"""
Inspect raw Parquet election files.

Outputs:
- data/manifest/raw_schema_report.csv
- data/preview/<dataset>_preview.csv
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
PREVIEW_DIR = PROJECT_ROOT / "data" / "preview"

SCHEMA_REPORT = MANIFEST_DIR / "raw_schema_report.csv"


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_DIR.rglob("*.parquet"))

    if not parquet_files:
        print(f"No Parquet files found in {RAW_DIR}")
        return

    rows = []

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)

            preview_path = PREVIEW_DIR / f"{file_path.stem}_preview.csv"
            df.head(100).to_csv(preview_path, index=False, encoding="utf-8-sig")

            rows.append(
                {
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "rows": len(df),
                    "columns_count": len(df.columns),
                    "columns": " | ".join(df.columns.astype(str)),
                    "dtypes": " | ".join(f"{col}: {dtype}" for col, dtype in df.dtypes.items()),
                    "preview_csv": str(preview_path.relative_to(PROJECT_ROOT)),
                }
            )

            print(f"[ok] {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")

        except Exception as exc:
            rows.append(
                {
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "rows": None,
                    "columns_count": None,
                    "columns": None,
                    "dtypes": None,
                    "preview_csv": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print(f"[error] {file_path.name}: {exc}")

    report = pd.DataFrame(rows)
    report.to_csv(SCHEMA_REPORT, index=False, encoding="utf-8-sig")

    print()
    print(f"Schema report written to: {SCHEMA_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"CSV previews written to: {PREVIEW_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
