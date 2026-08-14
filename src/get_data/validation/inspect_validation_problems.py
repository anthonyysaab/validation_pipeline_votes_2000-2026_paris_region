"""
Inspect validation problems after cleaning election results.

Reads:
- data/validation/suspicious_candidate_columns.csv
- data/validation/expressed_vote_mismatches.csv
- data/validation/per_booth_vote_checks.csv
- data/manifest/cleaning_report.csv
- data/clean/election_results_long.parquet
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

SUSPICIOUS_PATH = VALIDATION_DIR / "suspicious_candidate_columns.csv"
MISMATCHES_PATH = VALIDATION_DIR / "expressed_vote_mismatches.csv"
PER_BOOTH_PATH = VALIDATION_DIR / "per_booth_vote_checks.csv"
CLEANING_REPORT_PATH = MANIFEST_DIR / "cleaning_report.csv"
LONG_PATH = CLEAN_DIR / "election_results_long.parquet"


def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def read_csv_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a CSV report safely.

    Some validation reports can be genuinely empty when there are no problems.
    pandas.read_csv then raises EmptyDataError if the file has no header.
    In that case, return an empty DataFrame with expected columns.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def main() -> None:
    print_section("1. Suspicious candidate columns")

    if SUSPICIOUS_PATH.exists():
        suspicious = read_csv_or_empty(
            SUSPICIOUS_PATH,
            columns=["candidate", "reason"],
        )

        if suspicious.empty:
            print("No suspicious candidate columns.")
        else:
            print(suspicious.to_string(index=False))
    else:
        print(f"Missing file: {SUSPICIOUS_PATH}")

    print_section("2. Biggest expressed-vote mismatches")

    if MISMATCHES_PATH.exists():
        mismatches = read_csv_or_empty(MISMATCHES_PATH)

        if mismatches.empty:
            print("No mismatches.")
        else:
            cols = [
                "dataset_id",
                "id_bvote",
                "scrutin",
                "annee",
                "tour",
                "num_arrond",
                "num_bureau",
                "nb_exprim",
                "candidate_votes_sum",
                "expressed_vote_diff",
                "abs_expressed_vote_diff",
                "candidate_count",
            ]
            cols = [col for col in cols if col in mismatches.columns]

            print(mismatches[cols].head(80).to_string(index=False))

            if "dataset_id" in mismatches.columns:
                print()
                print("Mismatch count by dataset:")
                print(
                    mismatches.groupby("dataset_id")
                    .size()
                    .sort_values(ascending=False)
                    .to_string()
                )
    else:
        print(f"Missing file: {MISMATCHES_PATH}")

    print_section("3. Candidate columns detected per cleaned file")

    if CLEANING_REPORT_PATH.exists():
        report = read_csv_or_empty(CLEANING_REPORT_PATH)

        if report.empty:
            print("Cleaning report is empty.")
        else:
            cols = [
                "file",
                "status",
                "rows_raw",
                "candidate_columns",
                "rows_clean_long",
                "candidate_column_names",
                "reason",
            ]
            cols = [col for col in cols if col in report.columns]

            print(report[cols].to_string(index=False))
    else:
        print(f"Missing file: {CLEANING_REPORT_PATH}")

    print_section("4. Votes for suspicious candidates")

    if SUSPICIOUS_PATH.exists() and LONG_PATH.exists():
        suspicious = read_csv_or_empty(
            SUSPICIOUS_PATH,
            columns=["candidate", "reason"],
        )
        long_df = pd.read_parquet(LONG_PATH)

        if suspicious.empty:
            print("No suspicious candidates to inspect.")
        elif "candidate" not in suspicious.columns:
            print("Suspicious-candidate report has no 'candidate' column.")
        else:
            suspicious_names = suspicious["candidate"].dropna().astype(str).tolist()
            suspicious_votes = long_df[long_df["candidate"].isin(suspicious_names)]

            if suspicious_votes.empty:
                print("No matching suspicious candidates found in the clean long table.")
            else:
                summary = (
                    suspicious_votes.groupby(["dataset_id", "candidate"], dropna=False)
                    .agg(
                        rows=("votes", "size"),
                        total_votes=("votes", "sum"),
                        nonzero_rows=("votes", lambda x: int((x > 0).sum())),
                    )
                    .reset_index()
                    .sort_values(["dataset_id", "candidate"])
                )

                print(summary.to_string(index=False))
    else:
        print("Cannot inspect suspicious candidate votes because required files are missing.")


if __name__ == "__main__":
    main()