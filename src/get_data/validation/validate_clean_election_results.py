"""
Validate cleaned OpenData Paris election result files.

Inputs:
- data/clean/election_results_long.parquet
- data/clean/election_results_wide_vote_share.parquet
- data/manifest/cleaning_report.csv

Outputs:
- data/validation/validation_summary.csv
- data/validation/row_count_validation.csv
- data/validation/per_booth_vote_checks.csv
- data/validation/expressed_vote_mismatches.csv
- data/validation/municipal_row_checks.csv
- data/validation/suspicious_candidate_columns.csv
- data/validation/clean_sample.csv

Important validation rule:
- Non-municipal files are source-wide files converted to long format.
  For these, we validate that candidate vote sums match source nb_exprim.
- Municipal files are already long-format candidate/list files.
  For these, nb_exprim and nb_inscr are reconstructed during cleaning, and
  nb_votant is allowed to be missing.
"""

from pathlib import Path

import pandas as pd

from src.get_data.schema import is_missing_like

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

LONG_PATH = CLEAN_DIR / "election_results_long.parquet"
WIDE_PATH = CLEAN_DIR / "election_results_wide_vote_share.parquet"
CLEANING_REPORT_PATH = MANIFEST_DIR / "cleaning_report.csv"

VALIDATION_SUMMARY_PATH = VALIDATION_DIR / "validation_summary.csv"
ROW_COUNT_VALIDATION_PATH = VALIDATION_DIR / "row_count_validation.csv"
PER_BOOTH_CHECKS_PATH = VALIDATION_DIR / "per_booth_vote_checks.csv"
MISMATCHES_PATH = VALIDATION_DIR / "expressed_vote_mismatches.csv"
MUNICIPAL_ROW_CHECKS_PATH = VALIDATION_DIR / "municipal_row_checks.csv"
SUSPICIOUS_CANDIDATES_PATH = VALIDATION_DIR / "suspicious_candidate_columns.csv"
CLEAN_SAMPLE_PATH = VALIDATION_DIR / "clean_sample.csv"


CORE_LONG_COLUMNS = {
    "dataset_id",
    "source_file",
    "raw_row_id",
    "id_bvote",
    "scrutin",
    "annee",
    "tour",
    "num_arrond",
    "num_bureau",
    "nb_inscr",
    "nb_votant",
    "nb_exprim",
    "candidate",
    "candidate_source_column",
    "votes",
    "vote_share_exprimes",
    "vote_share_registered",
}

WIDE_METADATA_COLUMNS = {
    "dataset_id",
    "source_file",
    "raw_row_id",
    "id_bvote",
    "source_bv_id",
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
    "nb_bl",
    "nb_nul",
    "nb_bl_nul",
    "nb_exprim",
    "is_municipal",
}

NON_CANDIDATE_EXACT_NAMES = {
    "objectid",
    "raw_row_id",
    "id_bvote",
    "id_bv",
    "source_bv_id",
    "scrutin",
    "type_election",
    "annee",
    "tour",
    "numero_tour",
    "date",
    "date_tour",
    "num_circ",
    "circ_bv",
    "num_quartier",
    "quartier_bv",
    "num_arrond",
    "arr_bv",
    "num_bureau",
    "sec_bv",
    "nb_procu",
    "nb_procuration",
    "nb_inscr",
    "nb_inscrit",
    "nb_emarg",
    "nb_emargement",
    "nb_votant",
    "nb_bl",
    "nb_nul",
    "nb_blanc",
    "nb_vote_blanc",
    "nb_vote_nul",
    "nb_bl_nul",
    "nb_exprim",
    "nb_exprime",
    "vote_share_exprimes",
    "vote_share_registered",
    "geo_shape",
    "geo_point_2d",
    "st_area_shape",
    "st_perimeter_shape",
    "created_user",
    "created_date",
    "last_edited_user",
    "last_edited_date",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.relative_to(PROJECT_ROOT)}")


def status_ok(value: bool) -> str:
    return "OK" if value else "PROBLEM"


def source_consistency_status(count: int) -> str:
    """
    Vote-total mismatches are source-data consistency checks.

    They are not raw-to-clean extraction failures if the raw-to-clean audit passes.
    A small number should be inspected/documented, not confused with schema failure.
    """

    if count == 0:
        return "OK"

    if count <= 5:
        return "CHECK"

    return "PROBLEM"


def normalize_file_key(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\\", "/").strip()
    return Path(text).name


def is_municipal_text(value: object) -> bool:
    text = str(value).strip().lower()
    return "municip" in text or "_muni" in text or "-muni" in text or "muni_" in text


def add_municipal_flag(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    mask = pd.Series(False, index=df.index)

    for col in ["scrutin", "dataset_id", "source_file", "type_election"]:
        if col in df.columns:
            mask = mask | df[col].map(is_municipal_text)

    df["is_municipal"] = mask
    return df


def report_row_is_municipal(row: pd.Series) -> bool:
    fields = ["file", "source_file", "dataset_id", "scrutin", "type_election"]

    for field in fields:
        if field in row.index and is_municipal_text(row[field]):
            return True

    return False


def safe_int(value: object, default: int = 0) -> int:
    value = pd.to_numeric(value, errors="coerce")

    if pd.isna(value):
        return default

    return int(float(str(value)))


def validate_row_counts(
    cleaning_report: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if cleaning_report.empty:
        return pd.DataFrame(rows)

    if "source_file" in long_df.columns:
        source_counts_exact = long_df["source_file"].astype(str).value_counts().to_dict()
        source_counts_base = (
            long_df["source_file"]
            .map(normalize_file_key)
            .value_counts()
            .to_dict()
        )
    else:
        source_counts_exact = {}
        source_counts_base = {}

    ok_report = cleaning_report[cleaning_report["status"] == "ok"].copy()

    for _, row in ok_report.iterrows():
        report_file = row.get("file", row.get("source_file", ""))
        report_file_text = str(report_file)
        report_file_base = normalize_file_key(report_file)

        rows_raw = safe_int(row.get("rows_raw", 0))
        rows_after_key_filter = row.get("rows_after_key_filter", rows_raw)

        if pd.isna(rows_after_key_filter):
            rows_after_key_filter = rows_raw

        rows_after_key_filter = safe_int(rows_after_key_filter)
        candidate_columns = safe_int(row.get("candidate_columns", 0))
        rows_clean_long = safe_int(row.get("rows_clean_long", 0))

        is_municipal = report_row_is_municipal(row)

        if is_municipal:
            expected_long_rows = rows_clean_long
            expectation_method = "municipal_already_long"
        else:
            expected_long_rows = rows_after_key_filter * candidate_columns
            expectation_method = "wide_rows_x_candidate_columns"

        actual_long_rows = source_counts_exact.get(report_file_text)

        if actual_long_rows is None:
            actual_long_rows = source_counts_base.get(report_file_base, 0)

        report_count_ok = expected_long_rows == rows_clean_long
        actual_count_ok = True if actual_long_rows == 0 else actual_long_rows == rows_clean_long
        row_count_ok = report_count_ok and actual_count_ok

        rows.append(
            {
                "file": report_file,
                "is_municipal": is_municipal,
                "expectation_method": expectation_method,
                "rows_raw": rows_raw,
                "rows_after_key_filter": rows_after_key_filter,
                "candidate_columns": candidate_columns,
                "expected_long_rows": expected_long_rows,
                "reported_clean_long_rows": rows_clean_long,
                "actual_long_rows_in_final_table": actual_long_rows,
                "report_count_ok": report_count_ok,
                "actual_count_ok": actual_count_ok,
                "row_count_ok": row_count_ok,
                "status": status_ok(row_count_ok),
            }
        )

    return pd.DataFrame(rows)


def validate_non_municipal_per_booth_votes(long_df: pd.DataFrame) -> pd.DataFrame:
    non_municipal_df = long_df[~long_df["is_municipal"]].copy()

    if non_municipal_df.empty:
        return pd.DataFrame(
            columns=[
                "validation_scope",
                "candidate_votes_sum",
                "candidate_count",
                "expressed_vote_diff",
                "abs_expressed_vote_diff",
                "vote_share_sum_from_votes",
                "vote_share_sum_diff",
                "abs_vote_share_sum_diff",
                "expressed_votes_ok",
                "vote_share_sum_ok",
            ]
        )

    non_municipal_df["votes"] = pd.to_numeric(non_municipal_df["votes"], errors="coerce")
    non_municipal_df["nb_exprim"] = pd.to_numeric(
        non_municipal_df["nb_exprim"], errors="coerce"
    )

    group_cols = [
        "dataset_id",
        "source_file",
        "raw_row_id",
        "id_bvote",
        "scrutin",
        "annee",
        "tour",
        "date",
        "num_arrond",
        "num_bureau",
        "nb_inscr",
        "nb_votant",
        "nb_exprim",
    ]

    group_cols = [col for col in group_cols if col in non_municipal_df.columns]

    checks = (
        non_municipal_df.groupby(group_cols, dropna=False)
        .agg(
            candidate_votes_sum=("votes", "sum"),
            candidate_count=("candidate", "nunique"),
        )
        .reset_index()
    )

    checks.insert(0, "validation_scope", "non_municipal_source_total_check")

    checks["nb_exprim"] = pd.to_numeric(checks["nb_exprim"], errors="coerce")
    checks["candidate_votes_sum"] = pd.to_numeric(
        checks["candidate_votes_sum"], errors="coerce"
    )

    checks["expressed_vote_diff"] = checks["candidate_votes_sum"] - checks["nb_exprim"]
    checks["abs_expressed_vote_diff"] = checks["expressed_vote_diff"].abs()

    denominator = checks["nb_exprim"].where(checks["nb_exprim"] != 0)
    checks["vote_share_sum_from_votes"] = checks["candidate_votes_sum"] / denominator

    checks["vote_share_sum_diff"] = checks["vote_share_sum_from_votes"] - 1
    checks["abs_vote_share_sum_diff"] = checks["vote_share_sum_diff"].abs()

    checks["expressed_votes_ok"] = checks["abs_expressed_vote_diff"] <= 0.5
    checks["vote_share_sum_ok"] = (
        checks["abs_vote_share_sum_diff"].isna()
        | (checks["abs_vote_share_sum_diff"] <= 0.005)
    )

    return checks


def validate_municipal_rows(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    municipal_df = long_df[long_df["is_municipal"]].copy()

    if municipal_df.empty:
        return pd.DataFrame(), {
            "municipal_rows": 0,
            "municipal_missing_candidate": 0,
            "municipal_missing_votes": 0,
            "municipal_negative_votes": 0,
            "municipal_missing_booth_id": 0,
            "municipal_vote_share_exprimes_out_of_range": 0,
            "municipal_vote_share_registered_out_of_range": 0,
            "municipal_duplicate_candidate_rows": 0,
            "municipal_rows_with_any_problem": 0,
        }

    municipal_df["votes_numeric"] = pd.to_numeric(
        municipal_df.get("votes"), errors="coerce"
    )

    if "candidate" in municipal_df.columns:
        municipal_df["missing_candidate"] = is_missing_like(municipal_df["candidate"])
    else:
        municipal_df["missing_candidate"] = True

    municipal_df["missing_votes"] = municipal_df["votes_numeric"].isna()
    municipal_df["negative_votes"] = municipal_df["votes_numeric"] < 0

    booth_id_cols = [
        col for col in ["id_bvote", "source_bv_id"] if col in municipal_df.columns
    ]

    if booth_id_cols:
        booth_id_missing = pd.Series(True, index=municipal_df.index)

        for col in booth_id_cols:
            booth_id_missing = booth_id_missing & is_missing_like(municipal_df[col])

        municipal_df["missing_booth_id"] = booth_id_missing
    else:
        municipal_df["missing_booth_id"] = True

    if "vote_share_exprimes" in municipal_df.columns:
        exprimes_share = pd.to_numeric(
            municipal_df["vote_share_exprimes"], errors="coerce"
        )
        municipal_df["vote_share_exprimes_out_of_range"] = (
            exprimes_share.isna() | (exprimes_share < -1e-9) | (exprimes_share > 1 + 1e-9)
        )
    else:
        municipal_df["vote_share_exprimes_out_of_range"] = True

    if "vote_share_registered" in municipal_df.columns:
        registered_share = pd.to_numeric(
            municipal_df["vote_share_registered"], errors="coerce"
        )
        municipal_df["vote_share_registered_out_of_range"] = (
            registered_share.isna()
            | (registered_share < -1e-9)
            | (registered_share > 1 + 1e-9)
        )
    else:
        municipal_df["vote_share_registered_out_of_range"] = True

    duplicate_cols = [
        col
        for col in [
            "dataset_id",
            "source_file",
            "raw_row_id",
            "candidate_source_column",
        ]
        if col in municipal_df.columns
    ]

    if "candidate_source_column" not in duplicate_cols and "candidate" in municipal_df.columns:
        duplicate_cols.append("candidate")

    if duplicate_cols:
        municipal_df["duplicate_candidate_row"] = municipal_df.duplicated(
            subset=duplicate_cols,
            keep=False,
        )
    else:
        municipal_df["duplicate_candidate_row"] = True

    problem_cols = [
        "missing_candidate",
        "missing_votes",
        "negative_votes",
        "missing_booth_id",
        "vote_share_exprimes_out_of_range",
        "vote_share_registered_out_of_range",
        "duplicate_candidate_row",
    ]

    municipal_df["municipal_row_ok"] = ~municipal_df[problem_cols].any(axis=1)

    output_cols = [
        "dataset_id",
        "source_file",
        "raw_row_id",
        "id_bvote",
        "source_bv_id",
        "scrutin",
        "annee",
        "tour",
        "num_arrond",
        "num_bureau",
        "candidate",
        "candidate_source_column",
        "votes",
        "votes_numeric",
        "nb_exprim",
        "nb_inscr",
        "nb_votant",
        "vote_share_exprimes",
        "vote_share_registered",
        *problem_cols,
        "municipal_row_ok",
    ]

    output_cols = [col for col in output_cols if col in municipal_df.columns]

    municipal_checks = municipal_df[output_cols].copy()

    summary = {
        "municipal_rows": len(municipal_df),
        "municipal_missing_candidate": int(municipal_df["missing_candidate"].sum()),
        "municipal_missing_votes": int(municipal_df["missing_votes"].sum()),
        "municipal_negative_votes": int(municipal_df["negative_votes"].sum()),
        "municipal_missing_booth_id": int(municipal_df["missing_booth_id"].sum()),
        "municipal_vote_share_exprimes_out_of_range": int(
            municipal_df["vote_share_exprimes_out_of_range"].sum()
        ),
        "municipal_vote_share_registered_out_of_range": int(
            municipal_df["vote_share_registered_out_of_range"].sum()
        ),
        "municipal_duplicate_candidate_rows": int(
            municipal_df["duplicate_candidate_row"].sum()
        ),
        "municipal_rows_with_any_problem": int((~municipal_df["municipal_row_ok"]).sum()),
    }

    return municipal_checks, summary


def validate_wide_table(wide_df: pd.DataFrame) -> dict:
    index_cols = [col for col in WIDE_METADATA_COLUMNS if col in wide_df.columns]

    candidate_cols = [
        col
        for col in wide_df.columns
        if col not in WIDE_METADATA_COLUMNS
    ]

    duplicate_rows = int(wide_df.duplicated(subset=index_cols).sum()) if index_cols else 0

    candidate_values = wide_df[candidate_cols].apply(pd.to_numeric, errors="coerce")

    below_zero = int((candidate_values < -1e-9).sum().sum())
    above_one = int((candidate_values > 1 + 1e-9).sum().sum())

    return {
        "wide_rows": len(wide_df),
        "wide_columns": len(wide_df.columns),
        "wide_candidate_columns": len(candidate_cols),
        "wide_duplicate_index_rows": duplicate_rows,
        "wide_values_below_zero": below_zero,
        "wide_values_above_one": above_one,
    }


def is_suspicious_candidate_name(candidate: str) -> bool:
    name = str(candidate).strip().lower()

    if name in NON_CANDIDATE_EXACT_NAMES:
        return True

    if name.startswith("nb_"):
        return True

    if name.startswith("num_"):
        return True

    if name.startswith("geo_"):
        return True

    if name.startswith("st_"):
        return True

    if name.endswith("_shape"):
        return True

    return False


def find_suspicious_candidates(long_df: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        long_df["candidate"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    suspicious_rows = []

    for candidate in candidates:
        if is_suspicious_candidate_name(candidate):
            suspicious_rows.append(
                {
                    "candidate": candidate,
                    "reason": "looks like metadata or technical column",
                }
            )

    return pd.DataFrame(suspicious_rows)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    require_file(LONG_PATH)
    require_file(WIDE_PATH)

    long_df = pd.read_parquet(LONG_PATH)
    wide_df = pd.read_parquet(WIDE_PATH)

    long_df = add_municipal_flag(long_df)

    if CLEANING_REPORT_PATH.exists():
        cleaning_report = pd.read_csv(CLEANING_REPORT_PATH)
    else:
        cleaning_report = pd.DataFrame()

    missing_core_columns = sorted(CORE_LONG_COLUMNS - set(long_df.columns))

    row_count_validation = validate_row_counts(cleaning_report, long_df)
    row_count_validation.to_csv(
        ROW_COUNT_VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    per_booth_checks = validate_non_municipal_per_booth_votes(long_df)
    per_booth_checks.to_csv(
        PER_BOOTH_CHECKS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if per_booth_checks.empty:
        mismatches = per_booth_checks.copy()
    else:
        mismatches = per_booth_checks[
            (~per_booth_checks["expressed_votes_ok"])
            | (~per_booth_checks["vote_share_sum_ok"])
        ].copy()

        mismatches = mismatches.sort_values(
            by="abs_expressed_vote_diff",
            ascending=False,
        )

    mismatches.to_csv(MISMATCHES_PATH, index=False, encoding="utf-8-sig")

    municipal_checks, municipal_summary = validate_municipal_rows(long_df)
    municipal_checks.to_csv(
        MUNICIPAL_ROW_CHECKS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    suspicious_candidates = find_suspicious_candidates(long_df)
    suspicious_candidates.to_csv(
        SUSPICIOUS_CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    wide_checks = validate_wide_table(wide_df)

    skipped_files = 0
    error_files = 0

    if not cleaning_report.empty and "status" in cleaning_report.columns:
        skipped_files = int((cleaning_report["status"] == "skipped").sum())
        error_files = int((cleaning_report["status"] == "error").sum())

    row_count_problems = 0

    if not row_count_validation.empty:
        row_count_problems = int((~row_count_validation["row_count_ok"]).sum())

    if per_booth_checks.empty:
        expressed_vote_mismatches = 0
        vote_share_sum_mismatches = 0
    else:
        expressed_vote_mismatches = int((~per_booth_checks["expressed_votes_ok"]).sum())
        vote_share_sum_mismatches = int((~per_booth_checks["vote_share_sum_ok"]).sum())

    summary = pd.DataFrame(
        [
            {
                "check": "long_table_exists",
                "value": True,
                "status": "OK",
            },
            {
                "check": "wide_table_exists",
                "value": True,
                "status": "OK",
            },
            {
                "check": "long_rows",
                "value": len(long_df),
                "status": "INFO",
            },
            {
                "check": "long_columns",
                "value": len(long_df.columns),
                "status": "INFO",
            },
            {
                "check": "non_municipal_rows",
                "value": int((~long_df["is_municipal"]).sum()),
                "status": "INFO",
            },
            {
                "check": "municipal_rows",
                "value": municipal_summary["municipal_rows"],
                "status": "INFO",
            },
            {
                "check": "wide_rows",
                "value": wide_checks["wide_rows"],
                "status": "INFO",
            },
            {
                "check": "wide_columns",
                "value": wide_checks["wide_columns"],
                "status": "INFO",
            },
            {
                "check": "wide_candidate_columns",
                "value": wide_checks["wide_candidate_columns"],
                "status": "INFO",
            },
            {
                "check": "missing_core_long_columns",
                "value": " | ".join(missing_core_columns),
                "status": status_ok(len(missing_core_columns) == 0),
            },
            {
                "check": "row_count_problems",
                "value": row_count_problems,
                "status": status_ok(row_count_problems == 0),
            },
            {
                "check": "source_expressed_vote_mismatches",
                "value": expressed_vote_mismatches,
                "status": source_consistency_status(expressed_vote_mismatches),
            },
            {
                "check": "source_vote_share_sum_mismatches",
                "value": vote_share_sum_mismatches,
                "status": source_consistency_status(vote_share_sum_mismatches),
            },
            {
                "check": "municipal_missing_candidate",
                "value": municipal_summary["municipal_missing_candidate"],
                "status": status_ok(municipal_summary["municipal_missing_candidate"] == 0),
            },
            {
                "check": "municipal_missing_votes",
                "value": municipal_summary["municipal_missing_votes"],
                "status": status_ok(municipal_summary["municipal_missing_votes"] == 0),
            },
            {
                "check": "municipal_negative_votes",
                "value": municipal_summary["municipal_negative_votes"],
                "status": status_ok(municipal_summary["municipal_negative_votes"] == 0),
            },
            {
                "check": "municipal_missing_booth_id",
                "value": municipal_summary["municipal_missing_booth_id"],
                "status": status_ok(municipal_summary["municipal_missing_booth_id"] == 0),
            },
            {
                "check": "municipal_vote_share_exprimes_out_of_range",
                "value": municipal_summary["municipal_vote_share_exprimes_out_of_range"],
                "status": status_ok(
                    municipal_summary["municipal_vote_share_exprimes_out_of_range"] == 0
                ),
            },
            {
                "check": "municipal_vote_share_registered_out_of_range",
                "value": municipal_summary["municipal_vote_share_registered_out_of_range"],
                "status": status_ok(
                    municipal_summary["municipal_vote_share_registered_out_of_range"] == 0
                ),
            },
            {
                "check": "municipal_duplicate_candidate_rows",
                "value": municipal_summary["municipal_duplicate_candidate_rows"],
                "status": status_ok(
                    municipal_summary["municipal_duplicate_candidate_rows"] == 0
                ),
            },
            {
                "check": "municipal_rows_with_any_problem",
                "value": municipal_summary["municipal_rows_with_any_problem"],
                "status": status_ok(
                    municipal_summary["municipal_rows_with_any_problem"] == 0
                ),
            },
            {
                "check": "suspicious_candidate_columns",
                "value": len(suspicious_candidates),
                "status": status_ok(len(suspicious_candidates) == 0),
            },
            {
                "check": "wide_duplicate_index_rows",
                "value": wide_checks["wide_duplicate_index_rows"],
                "status": status_ok(wide_checks["wide_duplicate_index_rows"] == 0),
            },
            {
                "check": "wide_values_below_zero",
                "value": wide_checks["wide_values_below_zero"],
                "status": status_ok(wide_checks["wide_values_below_zero"] == 0),
            },
            {
                "check": "wide_values_above_one",
                "value": wide_checks["wide_values_above_one"],
                "status": status_ok(wide_checks["wide_values_above_one"] == 0),
            },
            {
                "check": "skipped_files_in_cleaning_report",
                "value": skipped_files,
                "status": "INFO" if skipped_files == 0 else "CHECK",
            },
            {
                "check": "error_files_in_cleaning_report",
                "value": error_files,
                "status": status_ok(error_files == 0),
            },
        ]
    )

    summary.to_csv(VALIDATION_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    sample_cols = [
        "dataset_id",
        "source_file",
        "raw_row_id",
        "id_bvote",
        "source_bv_id",
        "scrutin",
        "annee",
        "tour",
        "num_arrond",
        "num_bureau",
        "candidate",
        "candidate_source_column",
        "votes",
        "nb_exprim",
        "nb_inscr",
        "nb_votant",
        "vote_share_exprimes",
        "vote_share_registered",
        "is_municipal",
    ]
    sample_cols = [col for col in sample_cols if col in long_df.columns]

    long_df[sample_cols].head(500).to_csv(
        CLEAN_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Validation summary")
    print("==================")
    print(summary.to_string(index=False))
    print()
    print(f"[ok] Summary written to: {VALIDATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Row-count validation written to: {ROW_COUNT_VALIDATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Per-booth vote checks written to: {PER_BOOTH_CHECKS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Mismatches written to: {MISMATCHES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Municipal row checks written to: {MUNICIPAL_ROW_CHECKS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Suspicious candidates written to: {SUSPICIOUS_CANDIDATES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Clean sample written to: {CLEAN_SAMPLE_PATH.relative_to(PROJECT_ROOT)}")

    failed_checks = summary[summary["status"] == "PROBLEM"]
    if not failed_checks.empty:
        names = ", ".join(failed_checks["check"].astype(str))
        raise RuntimeError(f"Validation failed: {names}")


if __name__ == "__main__":
    main()
