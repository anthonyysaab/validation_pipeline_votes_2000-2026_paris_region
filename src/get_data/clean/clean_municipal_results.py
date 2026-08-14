"""
Clean municipal election result files.

Input:
- data/raw/elections-municipales-*.parquet

Outputs:
- data/clean/election_results_municipal_long.parquet
- data/manifest/cleaning_report_municipal.csv

Municipal files are already long-format:
one row = one polling station x one candidate/list.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.get_data.schema import is_missing_like, normalize_column_name, normalize_output_types

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"

LONG_OUTPUT = CLEAN_DIR / "election_results_municipal_long.parquet"
CLEANING_REPORT = MANIFEST_DIR / "cleaning_report_municipal.csv"


REQUIRED_COLUMNS = {
    "scrutin",
    "annee",
    "tour",
    "candidate",
    "votes",
    "source_bv_id",
}

STRING_COLUMNS = {
    "id_bvote",
    "source_bv_id",
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
    "nom",
    "prenom",
    "nuance",
    "liste",
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


FINAL_COLUMN_ORDER = [
    "id_bvote",
    "scrutin",
    "annee",
    "tour",
    "date",
    "num_circ",
    "num_quartier",
    "num_arrond",
    "num_bureau",
    "nb_procu",
    "nb_inscr",
    "nb_emarg",
    "nb_votant",
    "nb_bl_nul",
    "nb_exprim",
    "raw_row_id",
    "candidate_source_column",
    "votes",
    "candidate",
    "source_file",
    "dataset_id",
    "vote_share_exprimes",
    "vote_share_registered",
    "nb_bl",
    "nb_nul",
    "source_bv_id",
    "nom",
    "prenom",
    "nuance",
    "liste",
]


def replace_infinite_with_na(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(~np.isfinite(numeric), pd.NA)


def add_polling_station_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["source_bv_id"] = df["source_bv_id"].astype("string").str.strip()
    df["id_bvote"] = df["source_bv_id"]

    parsed = df["source_bv_id"].str.extract(
        r"^(?P<city_code>\d{5})_(?P<arrond>\d{2})(?P<bureau>\d{2})$"
    )

    df["num_arrond"] = parsed["arrond"].astype("string")
    df["num_bureau"] = parsed["bureau"].astype("string")

    return df


def add_reconstructed_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group_cols = ["dataset_id", "id_bvote"]

    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    df["nb_exprim"] = df.groupby(group_cols, dropna=False)["votes"].transform("sum")

    if "vote_share_registered" in df.columns:
        registered_share = pd.to_numeric(df["vote_share_registered"], errors="coerce")
        valid = registered_share.notna() & (registered_share > 0)

        registered_estimate = df["votes"].where(valid) / registered_share.where(valid)

        estimate_df = pd.DataFrame(
            {
                "dataset_id": df["dataset_id"],
                "id_bvote": df["id_bvote"],
                "registered_estimate": registered_estimate,
            }
        )

        station_registered = (
            estimate_df.dropna(subset=["registered_estimate"])
            .groupby(group_cols, dropna=False)["registered_estimate"]
            .median()
            .round()
            .rename("nb_inscr")
            .reset_index()
        )

        df = df.drop(columns=["nb_inscr"], errors="ignore").merge(
            station_registered,
            on=group_cols,
            how="left",
        )
    else:
        df["nb_inscr"] = pd.NA

    df["nb_votant"] = pd.NA
    df["nb_emarg"] = pd.NA
    df["nb_procu"] = pd.NA
    df["nb_bl"] = pd.NA
    df["nb_nul"] = pd.NA
    df["nb_bl_nul"] = pd.NA
    df["date"] = pd.NA
    df["num_circ"] = pd.NA
    df["num_quartier"] = pd.NA

    return df


def add_vote_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    votes = pd.to_numeric(df["votes"], errors="coerce")
    nb_exprim = pd.to_numeric(df["nb_exprim"], errors="coerce")
    nb_inscr = pd.to_numeric(df["nb_inscr"], errors="coerce")

    calculated_exprimes = replace_infinite_with_na(votes / nb_exprim)
    calculated_registered = replace_infinite_with_na(votes / nb_inscr)

    if "vote_share_exprimes" in df.columns:
        source_exprimes = replace_infinite_with_na(df["vote_share_exprimes"])
        df["vote_share_exprimes"] = source_exprimes.fillna(calculated_exprimes)
    else:
        df["vote_share_exprimes"] = calculated_exprimes

    if "vote_share_registered" in df.columns:
        source_registered = replace_infinite_with_na(df["vote_share_registered"])
        df["vote_share_registered"] = source_registered.fillna(calculated_registered)
    else:
        df["vote_share_registered"] = calculated_registered

    return df


def clean_one_file(file_path: Path) -> tuple[pd.DataFrame | None, dict]:
    df = pd.read_parquet(file_path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    df["raw_row_id"] = range(len(df))

    report = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "rows_raw": len(df),
        "columns_raw": len(df.columns),
        "status": "ok",
        "reason": "",
    }

    missing_required = REQUIRED_COLUMNS - set(df.columns)

    if missing_required:
        report["status"] = "skipped"
        report["reason"] = f"missing municipal required columns: {sorted(missing_required)}"
        return None, report

    df = add_polling_station_keys(df)

    df["dataset_id"] = file_path.stem
    df["source_file"] = file_path.name
    df["candidate_source_column"] = df["candidate"]

    before = len(df)

    id_ok = ~is_missing_like(df["id_bvote"])
    candidate_ok = ~is_missing_like(df["candidate"])
    votes_ok = pd.to_numeric(df["votes"], errors="coerce").notna()

    df = df[id_ok & candidate_ok & votes_ok].copy()

    report["rows_after_key_filter"] = len(df)
    report["rows_dropped_missing_keys"] = before - len(df)

    if df.empty:
        report["status"] = "skipped"
        report["reason"] = "no valid municipal rows after key filter"
        return None, report

    df = add_reconstructed_counts(df)
    df = add_vote_shares(df)

    existing_cols = [col for col in FINAL_COLUMN_ORDER if col in df.columns]
    extra_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + extra_cols].copy()

    df = normalize_output_types(df, STRING_COLUMNS, NUMERIC_COLUMNS)

    report["rows_clean_long"] = len(df)
    report["candidate_columns"] = 1
    report["candidate_column_names"] = "candidate"

    return df, report


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_DIR.glob("elections-municipales-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No municipal raw parquet files found in {RAW_DIR}")

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
        raise RuntimeError(f"{len(failed)} municipal file(s) failed or were skipped; see {CLEANING_REPORT}")

    if not cleaned_frames:
        raise RuntimeError("No municipal files could be cleaned.")

    long_df = pd.concat(cleaned_frames, ignore_index=True)
    long_df = normalize_output_types(long_df, STRING_COLUMNS, NUMERIC_COLUMNS)
    long_df.to_parquet(LONG_OUTPUT, index=False)

    print()
    print(f"[ok] Municipal long table written to: {LONG_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Cleaning report written to: {CLEANING_REPORT.relative_to(PROJECT_ROOT)}")
    print("Municipal long table shape:", long_df.shape)


if __name__ == "__main__":
    main()
