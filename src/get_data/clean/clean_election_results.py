"""
Clean non-municipal OpenData Paris election result files.

Input:
- data/raw/*.parquet, excluding elections-municipales-*.parquet

Outputs:
- data/clean/election_results_non_municipal_long.parquet
- data/manifest/cleaning_report_non_municipal.csv

This script handles the standard wide-format election files:
one row = one polling station, candidate vote counts are columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.get_data.schema import (
    STANDARD_COLUMNS,
    apply_column_aliases,
    candidate_columns_from_schema,
    is_missing_like,
    normalize_column_name,
    normalize_output_types,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"

LONG_OUTPUT = CLEAN_DIR / "election_results_non_municipal_long.parquet"
CLEANING_REPORT = MANIFEST_DIR / "cleaning_report_non_municipal.csv"


REQUIRED_RESULT_COLUMNS = {
    "id_bvote",
    "scrutin",
    "annee",
    "tour",
    "num_arrond",
    "num_bureau",
    "nb_inscr",
    "nb_votant",
    "nb_exprim",
}

SCRUTIN_CORRECTIONS = {
    "rÃ£Â©guinales": "RÃ©gionales",
    "rÃ©guinales": "RÃ©gionales",
    "regionales": "RÃ©gionales",
    "rÃ©gionales": "RÃ©gionales",
    "legislatives": "LÃ©gislative",
    "lÃ©gislatives": "LÃ©gislative",
    "presidentielles": "PrÃ©sidentielle",
    "prÃ©sidentielles": "PrÃ©sidentielle",
    "prÃ©sidentielle": "PrÃ©sidentielle",
    "europeennes": "EuropÃ©ennes",
    "europÃ©ennes": "EuropÃ©ennes",
}

STRING_COLUMNS = {
    "id_bvote",
    "scrutin",
    "tour",
    "num_circ",
    "num_quartier",
    "num_arrond",
    "num_bureau",
    "candidate",
    "candidate_source_column",
    "source_file",
    "dataset_id",
}

NUMERIC_COLUMNS = {
    "raw_row_id",
    "nb_procu",
    "nb_inscr",
    "nb_emarg",
    "nb_votant",
    "nb_bl",
    "nb_nul",
    "nb_bl_nul",
    "nb_exprim",
    "votes",
    "vote_share_exprimes",
    "vote_share_registered",
}


def normalize_scrutin_values(df: pd.DataFrame) -> pd.DataFrame:
    if "scrutin" not in df.columns:
        return df

    df = df.copy()

    def _fix(value):
        if pd.isna(value):
            return pd.NA

        text = str(value).strip()
        return SCRUTIN_CORRECTIONS.get(text.lower(), text)

    df["scrutin"] = df["scrutin"].apply(_fix)
    return df


def add_combined_blank_null_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "nb_bl_nul" not in df.columns and {"nb_bl", "nb_nul"}.issubset(df.columns):
        nb_bl = pd.to_numeric(df["nb_bl"], errors="coerce").fillna(0)
        nb_nul = pd.to_numeric(df["nb_nul"], errors="coerce").fillna(0)
        df["nb_bl_nul"] = nb_bl + nb_nul

    elif "nb_bl_nul" not in df.columns and "nb_bl" in df.columns:
        df["nb_bl_nul"] = pd.to_numeric(df["nb_bl"], errors="coerce").fillna(0)

    return df


def replace_infinite_with_na(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(~np.isfinite(numeric), pd.NA)


def drop_invalid_polling_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)

    id_ok = ~is_missing_like(df["id_bvote"])
    exprim_ok = pd.to_numeric(df["nb_exprim"], errors="coerce").notna()

    df = df[id_ok & exprim_ok].copy()

    dropped = before - len(df)
    return df, dropped


def clean_one_file(file_path: Path) -> tuple[pd.DataFrame | None, dict]:
    df = pd.read_parquet(file_path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    df["raw_row_id"] = range(len(df))

    df = apply_column_aliases(df)
    df = normalize_scrutin_values(df)
    df = add_combined_blank_null_column(df)

    report = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "rows_raw": len(df),
        "columns_raw": len(df.columns),
        "status": "ok",
        "reason": "",
    }

    missing_required = REQUIRED_RESULT_COLUMNS - set(df.columns)

    if missing_required:
        report["status"] = "skipped"
        report["reason"] = f"missing required columns: {sorted(missing_required)}"
        return None, report

    df, dropped_rows = drop_invalid_polling_rows(df)

    report["rows_after_key_filter"] = len(df)
    report["rows_dropped_missing_keys"] = dropped_rows

    metadata_cols = [col for col in df.columns if col in STANDARD_COLUMNS]
    candidate_cols = candidate_columns_from_schema(df, metadata_cols)

    if not candidate_cols:
        report["status"] = "skipped"
        report["reason"] = "no candidate columns detected"
        return None, report

    long_df = df.melt(
        id_vars=metadata_cols,
        value_vars=candidate_cols,
        var_name="candidate_source_column",
        value_name="votes",
    )

    long_df["candidate"] = long_df["candidate_source_column"]
    long_df["source_file"] = file_path.name
    long_df["dataset_id"] = file_path.stem

    long_df["votes"] = pd.to_numeric(long_df["votes"], errors="coerce").fillna(0)
    long_df["nb_exprim"] = pd.to_numeric(long_df["nb_exprim"], errors="coerce")
    long_df["nb_inscr"] = pd.to_numeric(long_df["nb_inscr"], errors="coerce")

    long_df["vote_share_exprimes"] = replace_infinite_with_na(
        long_df["votes"] / long_df["nb_exprim"]
    )
    long_df["vote_share_registered"] = replace_infinite_with_na(
        long_df["votes"] / long_df["nb_inscr"]
    )

    long_df = long_df[long_df["candidate"].notna()].copy()
    long_df = normalize_output_types(long_df, STRING_COLUMNS, NUMERIC_COLUMNS)

    report["rows_clean_long"] = len(long_df)
    report["candidate_columns"] = len(candidate_cols)
    report["candidate_column_names"] = " | ".join(candidate_cols)

    return long_df, report


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = [
        p
        for p in sorted(RAW_DIR.rglob("*.parquet"))
        if not p.name.startswith("elections-municipales-")
    ]

    if not parquet_files:
        raise FileNotFoundError(f"No non-municipal raw parquet files found in {RAW_DIR}")

    cleaned_frames = []
    reports = []

    for file_path in parquet_files:
        try:
            clean_df, report = clean_one_file(file_path)
            reports.append(report)

            if clean_df is not None:
                cleaned_frames.append(clean_df)
                print(f"[ok] {file_path.name}: {len(clean_df)} cleaned rows")
            else:
                print(f"[skip] {file_path.name}: {report['reason']}")

        except Exception as exc:
            reports.append(
                {
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "rows_raw": pd.NA,
                    "columns_raw": pd.NA,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[error] {file_path.name}: {exc}")

    report_df = pd.DataFrame(reports)
    report_df.to_csv(CLEANING_REPORT, index=False, encoding="utf-8-sig")

    failed = report_df[report_df["status"] != "ok"]
    if not failed.empty:
        raise RuntimeError(f"{len(failed)} non-municipal file(s) failed or were skipped; see {CLEANING_REPORT}")

    if not cleaned_frames:
        raise RuntimeError("No non-municipal election result files could be cleaned.")

    long_df = pd.concat(cleaned_frames, ignore_index=True)
    long_df = normalize_output_types(long_df, STRING_COLUMNS, NUMERIC_COLUMNS)
    long_df.to_parquet(LONG_OUTPUT, index=False)

    print()
    print(f"[ok] Non-municipal long table written to: {LONG_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Cleaning report written to: {CLEANING_REPORT.relative_to(PROJECT_ROOT)}")
    print("Non-municipal long table shape:", long_df.shape)


if __name__ == "__main__":
    main()
