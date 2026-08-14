"""
Audit raw-to-clean row correspondence.

This validates that every raw candidate vote case corresponds to exactly one
clean long-format row.

Inputs:
- data/raw/**/*.parquet
- data/clean/election_results_long.parquet

Outputs:
- data/audit/raw_to_clean_case_audit_summary.csv
- data/audit/raw_to_clean_case_audit_problems.csv

Supported raw shapes:

1. Non-municipal wide files
   - one row = one polling station
   - candidate vote counts are stored in candidate columns

2. Municipal long files
   - one row = one polling station x one candidate/list
   - candidate and votes are already explicit columns
"""

from pathlib import Path

import pandas as pd

from src.get_data.schema import (
    STANDARD_COLUMNS,
    apply_column_aliases,
    candidate_columns_from_schema,
    is_missing_like,
    normalize_column_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"

LONG_PATH = CLEAN_DIR / "election_results_long.parquet"

SUMMARY_OUTPUT = AUDIT_DIR / "raw_to_clean_case_audit_summary.csv"
PROBLEMS_OUTPUT = AUDIT_DIR / "raw_to_clean_case_audit_problems.csv"


REQUIRED_WIDE_RESULT_COLUMNS = {
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

REQUIRED_MUNICIPAL_COLUMNS = {
    "candidate",
    "votes",
    "source_bv_id",
}

def add_combined_blank_null_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "nb_bl_nul" not in df.columns and {"nb_bl", "nb_nul"}.issubset(df.columns):
        nb_bl = pd.to_numeric(df["nb_bl"], errors="coerce").fillna(0)
        nb_nul = pd.to_numeric(df["nb_nul"], errors="coerce").fillna(0)
        df["nb_bl_nul"] = nb_bl + nb_nul

    elif "nb_bl_nul" not in df.columns and "nb_bl" in df.columns:
        df["nb_bl_nul"] = pd.to_numeric(df["nb_bl"], errors="coerce").fillna(0)

    return df


def drop_invalid_wide_polling_rows(df: pd.DataFrame) -> pd.DataFrame:
    id_ok = ~is_missing_like(df["id_bvote"])
    exprim_ok = pd.to_numeric(df["nb_exprim"], errors="coerce").notna()

    return df[id_ok & exprim_ok].copy()


def drop_invalid_municipal_rows(df: pd.DataFrame) -> pd.DataFrame:
    source_ok = ~is_missing_like(df["source_bv_id"])
    candidate_ok = ~is_missing_like(df["candidate"])
    votes_ok = pd.to_numeric(df["votes"], errors="coerce").notna()

    return df[source_ok & candidate_ok & votes_ok].copy()


def looks_like_municipal_long_file(file_path: Path, df: pd.DataFrame) -> bool:
    if file_path.name.startswith("elections-municipales-"):
        return True

    return REQUIRED_MUNICIPAL_COLUMNS.issubset(df.columns)


def prepare_expected_municipal_cases(file_path: Path, df: pd.DataFrame) -> pd.DataFrame | None:
    missing = REQUIRED_MUNICIPAL_COLUMNS - set(df.columns)

    if missing:
        return None

    df = drop_invalid_municipal_rows(df)

    if df.empty:
        return None

    expected = pd.DataFrame(
        {
            "dataset_id": file_path.stem,
            "source_file": file_path.name,
            "raw_row_id": df["raw_row_id"],
            "candidate_source_column": df["candidate"],
            "expected_votes": pd.to_numeric(df["votes"], errors="coerce").fillna(0),
        }
    )

    return expected


def prepare_expected_wide_cases(file_path: Path, df: pd.DataFrame) -> pd.DataFrame | None:
    missing = REQUIRED_WIDE_RESULT_COLUMNS - set(df.columns)

    if missing:
        return None

    df = drop_invalid_wide_polling_rows(df)

    if df.empty:
        return None

    metadata_cols = [col for col in df.columns if col in STANDARD_COLUMNS]
    candidate_cols = candidate_columns_from_schema(df, metadata_cols)

    if not candidate_cols:
        return None

    expected = df.melt(
        id_vars=["raw_row_id"],
        value_vars=candidate_cols,
        var_name="candidate_source_column",
        value_name="expected_votes",
    )

    expected["source_file"] = file_path.name
    expected["dataset_id"] = file_path.stem
    expected["expected_votes"] = pd.to_numeric(
        expected["expected_votes"], errors="coerce"
    ).fillna(0)

    return expected[
        [
            "dataset_id",
            "source_file",
            "raw_row_id",
            "candidate_source_column",
            "expected_votes",
        ]
    ]


def prepare_expected_cases(file_path: Path) -> pd.DataFrame | None:
    df = pd.read_parquet(file_path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    df["raw_row_id"] = range(len(df))

    df = apply_column_aliases(df)
    df = add_combined_blank_null_column(df)

    if looks_like_municipal_long_file(file_path, df):
        return prepare_expected_municipal_cases(file_path, df)

    return prepare_expected_wide_cases(file_path, df)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if not LONG_PATH.exists():
        raise FileNotFoundError(f"Missing clean long file: {LONG_PATH}")

    clean_long = pd.read_parquet(LONG_PATH)

    required_clean_cols = {
        "dataset_id",
        "source_file",
        "raw_row_id",
        "candidate_source_column",
        "votes",
    }

    missing_clean_cols = required_clean_cols - set(clean_long.columns)

    if missing_clean_cols:
        raise ValueError(
            "Clean long table is missing required audit columns: "
            + ", ".join(sorted(missing_clean_cols))
        )

    clean_cases = clean_long[
        [
            "dataset_id",
            "source_file",
            "raw_row_id",
            "candidate_source_column",
            "votes",
        ]
    ].copy()

    clean_cases["dataset_id"] = clean_cases["dataset_id"].astype("string")
    clean_cases["source_file"] = clean_cases["source_file"].astype("string")
    clean_cases["candidate_source_column"] = clean_cases[
        "candidate_source_column"
    ].astype("string")

    clean_cases["raw_row_id"] = pd.to_numeric(
        clean_cases["raw_row_id"], errors="coerce"
    ).astype("Int64")
    clean_cases["votes"] = pd.to_numeric(clean_cases["votes"], errors="coerce").fillna(0)

    raw_files = sorted(RAW_DIR.rglob("*.parquet"))

    expected_frames = []
    skipped_files = []

    for file_path in raw_files:
        expected = prepare_expected_cases(file_path)

        if expected is None:
            skipped_files.append(str(file_path.relative_to(PROJECT_ROOT)))
            continue

        expected_frames.append(expected)

    if not expected_frames:
        raise ValueError("No expected raw cases could be built from raw Parquet files.")

    expected_cases = pd.concat(expected_frames, ignore_index=True)

    expected_cases["dataset_id"] = expected_cases["dataset_id"].astype("string")
    expected_cases["source_file"] = expected_cases["source_file"].astype("string")
    expected_cases["candidate_source_column"] = expected_cases[
        "candidate_source_column"
    ].astype("string")

    expected_cases["raw_row_id"] = pd.to_numeric(
        expected_cases["raw_row_id"], errors="coerce"
    ).astype("Int64")
    expected_cases["expected_votes"] = pd.to_numeric(
        expected_cases["expected_votes"], errors="coerce"
    ).fillna(0)

    audit = expected_cases.merge(
        clean_cases,
        on=["dataset_id", "source_file", "raw_row_id", "candidate_source_column"],
        how="outer",
        indicator=True,
    )

    audit["expected_votes"] = pd.to_numeric(
        audit["expected_votes"], errors="coerce"
    ).fillna(0)
    audit["votes"] = pd.to_numeric(audit["votes"], errors="coerce").fillna(0)
    audit["vote_diff"] = audit["votes"] - audit["expected_votes"]
    audit["abs_vote_diff"] = audit["vote_diff"].abs()

    missing_from_clean = audit[audit["_merge"] == "left_only"].copy()
    extra_in_clean = audit[audit["_merge"] == "right_only"].copy()
    vote_mismatches = audit[
        (audit["_merge"] == "both") & (audit["abs_vote_diff"] > 0.000001)
    ].copy()

    duplicate_clean_cases = (
        clean_cases.groupby(
            ["dataset_id", "source_file", "raw_row_id", "candidate_source_column"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    duplicate_clean_cases = duplicate_clean_cases[duplicate_clean_cases["count"] > 1]

    duplicate_expected_cases = (
        expected_cases.groupby(
            ["dataset_id", "source_file", "raw_row_id", "candidate_source_column"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    duplicate_expected_cases = duplicate_expected_cases[
        duplicate_expected_cases["count"] > 1
    ]

    problem_frames = []

    if not missing_from_clean.empty:
        tmp = missing_from_clean.copy()
        tmp["problem_type"] = "missing_from_clean"
        problem_frames.append(tmp)

    if not extra_in_clean.empty:
        tmp = extra_in_clean.copy()
        tmp["problem_type"] = "extra_in_clean"
        problem_frames.append(tmp)

    if not vote_mismatches.empty:
        tmp = vote_mismatches.copy()
        tmp["problem_type"] = "vote_value_mismatch"
        problem_frames.append(tmp)

    if not duplicate_clean_cases.empty:
        tmp = duplicate_clean_cases.copy()
        tmp["problem_type"] = "duplicate_clean_case"
        problem_frames.append(tmp)

    if not duplicate_expected_cases.empty:
        tmp = duplicate_expected_cases.copy()
        tmp["problem_type"] = "duplicate_expected_case"
        problem_frames.append(tmp)

    if problem_frames:
        problems = pd.concat(problem_frames, ignore_index=True, sort=False)
    else:
        problems = pd.DataFrame(
            columns=[
                "problem_type",
                "dataset_id",
                "source_file",
                "raw_row_id",
                "candidate_source_column",
                "expected_votes",
                "votes",
                "vote_diff",
            ]
        )

    summary = pd.DataFrame(
        [
            {
                "check": "raw_files_used_for_expected_cases",
                "value": len(expected_frames),
                "status": "OK" if len(expected_frames) > 0 else "PROBLEM",
            },
            {
                "check": "raw_files_skipped_for_expected_cases",
                "value": len(skipped_files),
                "status": "OK" if len(skipped_files) == 0 else "CHECK",
            },
            {
                "check": "expected_raw_candidate_cases",
                "value": len(expected_cases),
                "status": "INFO",
            },
            {
                "check": "clean_long_cases",
                "value": len(clean_cases),
                "status": "INFO",
            },
            {
                "check": "missing_from_clean",
                "value": len(missing_from_clean),
                "status": "OK" if len(missing_from_clean) == 0 else "PROBLEM",
            },
            {
                "check": "extra_in_clean",
                "value": len(extra_in_clean),
                "status": "OK" if len(extra_in_clean) == 0 else "PROBLEM",
            },
            {
                "check": "vote_value_mismatches",
                "value": len(vote_mismatches),
                "status": "OK" if len(vote_mismatches) == 0 else "PROBLEM",
            },
            {
                "check": "duplicate_clean_cases",
                "value": len(duplicate_clean_cases),
                "status": "OK" if len(duplicate_clean_cases) == 0 else "PROBLEM",
            },
            {
                "check": "duplicate_expected_cases",
                "value": len(duplicate_expected_cases),
                "status": "OK" if len(duplicate_expected_cases) == 0 else "PROBLEM",
            },
        ]
    )

    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    problems.to_csv(PROBLEMS_OUTPUT, index=False, encoding="utf-8-sig")

    print()
    print("Raw-to-clean case audit summary")
    print("===============================")
    print(summary.to_string(index=False))
    print()
    print(f"[ok] Summary written to: {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Problems written to: {PROBLEMS_OUTPUT.relative_to(PROJECT_ROOT)}")

    if skipped_files:
        print()
        print("[warn] Raw files skipped while building expected cases:")
        for file_name in skipped_files:
            print("  -", file_name)

    failed_checks = summary[summary["status"] == "PROBLEM"]
    if not failed_checks.empty:
        names = ", ".join(failed_checks["check"].astype(str))
        raise RuntimeError(f"Raw-to-clean audit failed: {names}")


if __name__ == "__main__":
    main()
