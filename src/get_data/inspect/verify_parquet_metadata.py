"""
Verify Parquet metadata and schema consistency.

This script checks:
- raw Parquet file-level metadata;
- raw Parquet column-level metadata;
- whether raw election files have the required columns after alias normalization;
- whether numeric candidate columns are really numeric in the Parquet schema;
- whether geometry / technical metadata columns are correctly excluded from candidates;
- whether clean output files have stable expected column types.

Inputs:
- data/raw/**/*.parquet
- data/clean/election_results_long.parquet
- data/clean/election_results_wide_vote_share.parquet

Outputs:
- data/metadata/parquet_file_metadata.csv
- data/metadata/parquet_column_metadata.csv
- data/metadata/metadata_validation_summary.csv
- data/metadata/metadata_validation_problems.csv
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.get_data.schema import (
    COLUMN_ALIASES,
    NON_CANDIDATE_COLUMNS,
    STANDARD_COLUMNS,
    normalize_column_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

LONG_PATH = CLEAN_DIR / "election_results_long.parquet"
WIDE_PATH = CLEAN_DIR / "election_results_wide_vote_share.parquet"

FILE_METADATA_OUTPUT = METADATA_DIR / "parquet_file_metadata.csv"
COLUMN_METADATA_OUTPUT = METADATA_DIR / "parquet_column_metadata.csv"
VALIDATION_SUMMARY_OUTPUT = METADATA_DIR / "metadata_validation_summary.csv"
VALIDATION_PROBLEMS_OUTPUT = METADATA_DIR / "metadata_validation_problems.csv"


REQUIRED_WIDE_RAW_COLUMNS = {
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

REQUIRED_MUNICIPAL_RAW_COLUMNS = {
    "scrutin",
    "annee",
    "tour",
    "candidate",
    "votes",
    "source_bv_id",
}

EXPECTED_CLEAN_LONG_COLUMNS = {
    "dataset_id",
    "source_file",
    "id_bvote",
    "scrutin",
    "annee",
    "tour",
    "candidate",
    "votes",
    "nb_exprim",
    "vote_share_exprimes",
    "vote_share_registered",
}

EXPECTED_CLEAN_LONG_TYPES = {
    "dataset_id": "string",
    "source_file": "string",
    "id_bvote": "string",
    "scrutin": "string",
    "annee": "int64",
    "tour": "string",
    "candidate": "string",
    "votes": "double",
    "nb_exprim": "double",
    "vote_share_exprimes": "double",
    "vote_share_registered": "double",
}

CLEAN_WIDE_REQUIRED_METADATA_COLUMNS = {
    "dataset_id",
    "source_file",
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

CLEAN_WIDE_METADATA_COLUMNS = CLEAN_WIDE_REQUIRED_METADATA_COLUMNS | {
    "source_bv_id",
    "date",
    "num_circ",
    "num_quartier",
}


def canonical_column_name(col: str) -> str:
    normalized = normalize_column_name(col)
    return COLUMN_ALIASES.get(normalized, normalized)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def is_municipal_raw_file(file_path: Path) -> bool:
    return file_path.name.startswith("elections-municipales-")


def required_columns_for_raw_file(file_path: Path) -> set[str]:
    """
    Municipal raw files are already long-format candidate/list files.

    They do not contain nb_exprim, nb_inscr, or nb_votant in the raw source.
    Those fields are reconstructed later by clean_municipal_results.py.
    """
    if is_municipal_raw_file(file_path):
        return REQUIRED_MUNICIPAL_RAW_COLUMNS

    return REQUIRED_WIDE_RAW_COLUMNS


def decode_metadata(metadata: dict[bytes, bytes] | None) -> str:
    if not metadata:
        return ""

    decoded = {}

    for key, value in metadata.items():
        key_text = key.decode("utf-8", errors="replace")
        value_text = value.decode("utf-8", errors="replace")
        decoded[key_text] = value_text

    return json.dumps(decoded, ensure_ascii=False)


def schema_hash(schema: pa.Schema) -> str:
    schema_text = str(schema)
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:16]


def is_numeric_arrow_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
    )


def is_string_like_arrow_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_dictionary(data_type)
    )


def classify_column_role(original_name: str, arrow_type: pa.DataType) -> str:
    canonical = canonical_column_name(original_name)

    if canonical in STANDARD_COLUMNS:
        return "standard_metadata"

    if canonical in NON_CANDIDATE_COLUMNS:
        return "technical_or_non_candidate"

    if is_numeric_arrow_type(arrow_type):
        return "candidate_vote_column"

    if is_string_like_arrow_type(arrow_type):
        return "string_non_candidate_or_unknown"

    return "unknown_or_complex"


def read_parquet_metadata(file_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parquet_file = pq.ParquetFile(file_path)
    parquet_metadata = parquet_file.metadata
    arrow_schema = parquet_file.schema_arrow

    file_row = {
        "file": relative(file_path),
        "num_rows_metadata": parquet_metadata.num_rows,
        "num_columns_metadata": parquet_metadata.num_columns,
        "num_row_groups": parquet_metadata.num_row_groups,
        "created_by": parquet_metadata.created_by,
        "format_version": parquet_metadata.format_version,
        "serialized_size": parquet_metadata.serialized_size,
        "schema_hash": schema_hash(arrow_schema),
        "key_value_metadata": decode_metadata(parquet_metadata.metadata),
    }

    column_rows = []

    for index, field in enumerate(arrow_schema):
        original_name = field.name
        canonical_name = canonical_column_name(original_name)
        arrow_type = field.type
        role = classify_column_role(original_name, arrow_type)

        try:
            parquet_column = parquet_metadata.schema.column(index)
            physical_type = str(parquet_column.physical_type)
            logical_type = str(parquet_column.logical_type)
            converted_type = str(parquet_column.converted_type)
        except Exception:
            physical_type = ""
            logical_type = ""
            converted_type = ""

        column_rows.append(
            {
                "file": relative(file_path),
                "column_index": index,
                "original_column": original_name,
                "canonical_column": canonical_name,
                "arrow_type": str(arrow_type),
                "nullable": field.nullable,
                "parquet_physical_type": physical_type,
                "parquet_logical_type": logical_type,
                "parquet_converted_type": converted_type,
                "in_standard_columns": canonical_name in STANDARD_COLUMNS,
                "in_required_wide_raw_columns": canonical_name in REQUIRED_WIDE_RAW_COLUMNS,
                "in_required_municipal_raw_columns": canonical_name in REQUIRED_MUNICIPAL_RAW_COLUMNS,
                "in_non_candidate_columns": canonical_name in NON_CANDIDATE_COLUMNS,
                "is_numeric_arrow_type": is_numeric_arrow_type(arrow_type),
                "detected_role": role,
            }
        )

    return file_row, column_rows


def validate_raw_file(file_path: Path, column_metadata: pd.DataFrame) -> list[dict[str, Any]]:
    problems = []

    current = column_metadata[column_metadata["file"] == relative(file_path)].copy()
    canonical_columns = set(current["canonical_column"].astype(str))

    required_columns = required_columns_for_raw_file(file_path)
    missing_required = sorted(required_columns - canonical_columns)

    if missing_required:
        problems.append(
            {
                "file": relative(file_path),
                "check": "raw_required_columns_after_alias",
                "status": "PROBLEM",
                "details": "missing required columns: " + " | ".join(missing_required),
            }
        )

    candidate_rows = current[current["detected_role"] == "candidate_vote_column"].copy()

    non_numeric_candidates = candidate_rows[
        candidate_rows["is_numeric_arrow_type"].ne(True)
    ]

    if not non_numeric_candidates.empty:
        problems.append(
            {
                "file": relative(file_path),
                "check": "candidate_columns_are_numeric",
                "status": "PROBLEM",
                "details": "non-numeric candidates: "
                + " | ".join(non_numeric_candidates["original_column"].astype(str).tolist()),
            }
        )

    accidentally_candidate_non_candidate = current[
        (current["canonical_column"].isin(NON_CANDIDATE_COLUMNS))
        & (current["detected_role"] == "candidate_vote_column")
    ]

    if not accidentally_candidate_non_candidate.empty:
        problems.append(
            {
                "file": relative(file_path),
                "check": "technical_columns_not_candidates",
                "status": "PROBLEM",
                "details": "technical columns detected as candidate: "
                + " | ".join(
                    accidentally_candidate_non_candidate["original_column"].astype(str).tolist()
                ),
            }
        )

    candidate_count = len(candidate_rows)

    if candidate_count == 0:
        problems.append(
            {
                "file": relative(file_path),
                "check": "candidate_column_count",
                "status": "PROBLEM",
                "details": "no numeric candidate/vote columns detected from metadata",
            }
        )

    return problems


def validate_clean_long_schema() -> list[dict[str, Any]]:
    problems = []

    if not LONG_PATH.exists():
        return [
            {
                "file": relative(LONG_PATH),
                "check": "clean_long_exists",
                "status": "PROBLEM",
                "details": "missing clean long Parquet output",
            }
        ]

    schema = pq.read_schema(LONG_PATH)
    columns = set(schema.names)

    missing = sorted(EXPECTED_CLEAN_LONG_COLUMNS - columns)

    if missing:
        problems.append(
            {
                "file": relative(LONG_PATH),
                "check": "clean_long_expected_columns",
                "status": "PROBLEM",
                "details": "missing columns: " + " | ".join(missing),
            }
        )

    for col, expected_type_fragment in EXPECTED_CLEAN_LONG_TYPES.items():
        if col not in schema.names:
            continue

        actual_type = str(schema.field(col).type).lower()

        if expected_type_fragment not in actual_type:
            problems.append(
                {
                    "file": relative(LONG_PATH),
                    "check": "clean_long_column_type",
                    "status": "PROBLEM",
                    "details": f"{col}: expected type containing '{expected_type_fragment}', got '{actual_type}'",
                }
            )

    return problems


def validate_clean_wide_schema() -> list[dict[str, Any]]:
    problems = []

    if not WIDE_PATH.exists():
        return [
            {
                "file": relative(WIDE_PATH),
                "check": "clean_wide_exists",
                "status": "PROBLEM",
                "details": "missing clean wide Parquet output",
            }
        ]

    schema = pq.read_schema(WIDE_PATH)
    columns = set(schema.names)

    missing = sorted(CLEAN_WIDE_REQUIRED_METADATA_COLUMNS - columns)

    if missing:
        problems.append(
            {
                "file": relative(WIDE_PATH),
                "check": "clean_wide_metadata_columns",
                "status": "PROBLEM",
                "details": "missing columns: " + " | ".join(missing),
            }
        )

    candidate_cols = [
        col for col in schema.names
        if col not in CLEAN_WIDE_METADATA_COLUMNS
    ]

    non_numeric_vote_share_cols = []

    for col in candidate_cols:
        arrow_type = schema.field(col).type

        if not is_numeric_arrow_type(arrow_type):
            non_numeric_vote_share_cols.append(f"{col}: {arrow_type}")

    if non_numeric_vote_share_cols:
        problems.append(
            {
                "file": relative(WIDE_PATH),
                "check": "clean_wide_candidate_vote_share_types",
                "status": "PROBLEM",
                "details": "non-numeric wide candidate columns: "
                + " | ".join(non_numeric_vote_share_cols),
            }
        )

    return problems


def build_summary(
    raw_files: list[Path],
    file_metadata: pd.DataFrame,
    column_metadata: pd.DataFrame,
    problems: pd.DataFrame,
) -> pd.DataFrame:
    if problems.empty:
        problem_count = 0
    else:
        problem_count = len(problems)

    raw_file_names = {relative(path) for path in raw_files}
    raw_column_metadata = column_metadata[
        column_metadata["file"].isin(raw_file_names)
    ].copy()

    raw_candidate_columns = raw_column_metadata[
        raw_column_metadata["detected_role"] == "candidate_vote_column"
    ]

    technical_candidate_errors = raw_column_metadata[
        (raw_column_metadata["canonical_column"].isin(NON_CANDIDATE_COLUMNS))
        & (raw_column_metadata["detected_role"] == "candidate_vote_column")
    ]

    expected_file_metadata_rows = (
        len(raw_files)
        + int(LONG_PATH.exists())
        + int(WIDE_PATH.exists())
    )

    summary_rows = [
        {
            "check": "raw_parquet_files_found",
            "value": len(raw_files),
            "status": "OK" if len(raw_files) > 0 else "PROBLEM",
        },
        {
            "check": "file_metadata_rows",
            "value": len(file_metadata),
            "status": "OK" if len(file_metadata) == expected_file_metadata_rows else "CHECK",
        },
        {
            "check": "column_metadata_rows",
            "value": len(column_metadata),
            "status": "OK" if len(column_metadata) > 0 else "PROBLEM",
        },
        {
            "check": "raw_candidate_vote_columns_detected",
            "value": len(raw_candidate_columns),
            "status": "OK" if len(raw_candidate_columns) > 0 else "PROBLEM",
        },
        {
            "check": "technical_columns_detected_as_candidates",
            "value": len(technical_candidate_errors),
            "status": "OK" if len(technical_candidate_errors) == 0 else "PROBLEM",
        },
        {
            "check": "metadata_validation_problems",
            "value": problem_count,
            "status": "OK" if problem_count == 0 else "PROBLEM",
        },
    ]

    return pd.DataFrame(summary_rows)


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.rglob("*.parquet"))

    files_to_check = raw_files.copy()

    if LONG_PATH.exists():
        files_to_check.append(LONG_PATH)

    if WIDE_PATH.exists():
        files_to_check.append(WIDE_PATH)

    if not files_to_check:
        print("No Parquet files found.")
        return

    file_rows = []
    column_rows = []
    read_errors = []

    for file_path in files_to_check:
        try:
            file_row, current_column_rows = read_parquet_metadata(file_path)
            file_rows.append(file_row)
            column_rows.extend(current_column_rows)
            print(f"[ok] metadata read: {relative(file_path)}")

        except Exception as exc:
            read_errors.append(
                {
                    "file": relative(file_path),
                    "check": "read_parquet_metadata",
                    "status": "PROBLEM",
                    "details": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[error] metadata read failed: {relative(file_path)}: {exc}")

    file_metadata = pd.DataFrame(file_rows)
    column_metadata = pd.DataFrame(column_rows)

    validation_problems = read_errors.copy()

    for raw_file in raw_files:
        validation_problems.extend(validate_raw_file(raw_file, column_metadata))

    validation_problems.extend(validate_clean_long_schema())
    validation_problems.extend(validate_clean_wide_schema())

    problems_df = pd.DataFrame(validation_problems)

    if problems_df.empty:
        problems_df = pd.DataFrame(columns=["file", "check", "status", "details"])

    summary = build_summary(raw_files, file_metadata, column_metadata, problems_df)

    file_metadata.to_csv(FILE_METADATA_OUTPUT, index=False, encoding="utf-8-sig")
    column_metadata.to_csv(COLUMN_METADATA_OUTPUT, index=False, encoding="utf-8-sig")
    problems_df.to_csv(VALIDATION_PROBLEMS_OUTPUT, index=False, encoding="utf-8-sig")
    summary.to_csv(VALIDATION_SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    print()
    print("Metadata validation summary")
    print("===========================")
    print(summary.to_string(index=False))

    print()
    print(f"[ok] File metadata written to: {relative(FILE_METADATA_OUTPUT)}")
    print(f"[ok] Column metadata written to: {relative(COLUMN_METADATA_OUTPUT)}")
    print(f"[ok] Validation summary written to: {relative(VALIDATION_SUMMARY_OUTPUT)}")
    print(f"[ok] Validation problems written to: {relative(VALIDATION_PROBLEMS_OUTPUT)}")

    if not problems_df.empty:
        print()
        print("Problems detected:")
        print(problems_df.to_string(index=False))


if __name__ == "__main__":
    main()
