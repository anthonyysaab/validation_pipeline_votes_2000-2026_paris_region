"""
Export cleaned Paris election Parquet datasets to CSV.

This is mainly for teammates who prefer CSV/Excel workflows.
Parquet remains the preferred format for Python analysis.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CSV_DIR = CLEAN_DIR / "csv"
CSV_BY_ELECTION_DIR = CSV_DIR / "by_election"

PARQUET_EXPORTS = [
    "election_results_long",
    "election_results_wide_vote_share",
    "election_results_non_municipal_long",
    "election_results_municipal_long",
]


def safe_filename(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def export_main_csv_files() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    for name in PARQUET_EXPORTS:
        input_path = CLEAN_DIR / f"{name}.parquet"
        output_path = CSV_DIR / f"{name}.csv"

        if not input_path.exists():
            print(f"[skip] Missing: {input_path.relative_to(PROJECT_ROOT)}")
            continue

        df = pd.read_parquet(input_path)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(
            f"[ok] Exported {len(df):,} rows -> "
            f"{output_path.relative_to(PROJECT_ROOT)}"
        )


def export_long_csv_by_election() -> None:
    input_path = CLEAN_DIR / "election_results_long.parquet"

    if not input_path.exists():
        print(f"[skip] Missing: {input_path.relative_to(PROJECT_ROOT)}")
        return

    CSV_BY_ELECTION_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    for dataset_id, group in df.groupby("dataset_id", dropna=False):
        filename = safe_filename(dataset_id)
        output_path = CSV_BY_ELECTION_DIR / f"{filename}.csv"

        group.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(
            f"[ok] Exported {len(group):,} rows -> "
            f"{output_path.relative_to(PROJECT_ROOT)}"
        )


def main() -> None:
    export_main_csv_files()
    export_long_csv_by_election()


if __name__ == "__main__":
    main()
