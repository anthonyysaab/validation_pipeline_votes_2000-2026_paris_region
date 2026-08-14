"""Shared source-schema and normalization rules for election data."""

import re

import pandas as pd
from pandas.api.types import is_numeric_dtype

MISSING_TEXT_VALUES = {"", "nan", "none", "<na>", "null"}

STANDARD_COLUMNS = {
    "raw_row_id", "id_bvote", "source_bv_id", "dataset_id", "source_file",
    "scrutin", "annee", "tour",
    "date", "num_circ", "num_quartier", "num_arrond", "num_bureau",
    "nb_procu", "nb_inscr", "nb_emarg", "nb_votant", "nb_bl", "nb_nul",
    "nb_bl_nul", "nb_exprim",
}

COLUMN_ALIASES = {
    "id_bv": "id_bvote",
    "type_election": "scrutin",
    "numero_tour": "tour",
    "date_tour": "date",
    "circ_bv": "num_circ",
    "quartier_bv": "num_quartier",
    "arr_bv": "num_arrond",
    "nb_procuration": "nb_procu",
    "nb_inscrit": "nb_inscr",
    "nb_emargement": "nb_emarg",
    "nb_exprime": "nb_exprim",
    "nb_vote_blanc": "nb_bl",
    "nb_vote_nul": "nb_nul",
    "nb_blanc": "nb_bl",
}

NON_CANDIDATE_COLUMNS = {
    "objectid", "geo_shape", "geo_point_2d", "st_area_shape",
    "st_perimeter_shape", "created_user", "created_date", "last_edited_user",
    "last_edited_date", "nb_bl", "nb_nul", "nb_blanc", "nb_vote_blanc",
    "nb_vote_nul", "sec_bv",
}


def normalize_column_name(column: object) -> str:
    return str(column).strip().lower()


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        old: new
        for old, new in COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(columns=rename_map)


def is_missing_like(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isna() | text.isin(MISSING_TEXT_VALUES)


def extract_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(20\d{2})", str(value))
    return int(match.group(1)) if match else None


def normalize_output_types(
    df: pd.DataFrame,
    string_columns: set[str],
    numeric_columns: set[str],
) -> pd.DataFrame:
    df = df.copy()
    if "annee" in df.columns:
        df["annee"] = df["annee"].apply(extract_year).astype("Int64")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df["date"] = dates.dt.date.astype("string")
    for column in string_columns & set(df.columns):
        df[column] = df[column].astype("string")
    for column in numeric_columns & set(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


def candidate_columns_from_schema(
    df: pd.DataFrame,
    metadata_columns: list[str],
) -> list[str]:
    """Classify every non-metadata source column using its Parquet-backed dtype."""
    candidates = []
    unknown = []
    excluded = set(metadata_columns) | NON_CANDIDATE_COLUMNS

    for column in df.columns:
        if column in excluded:
            continue
        if is_numeric_dtype(df[column].dtype):
            candidates.append(column)
        else:
            unknown.append(column)

    if unknown:
        raise ValueError("Unclassified non-numeric source columns: " + ", ".join(unknown))
    return candidates
