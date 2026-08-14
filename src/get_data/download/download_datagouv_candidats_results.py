"""
download_datagouv_candidats_results.py

Download the official data.gouv.fr aggregated candidate-level election results
parquet file and save it under the local path expected by the municipal
extraction step.

This removes the hidden dependency on a local-only intermediate file:

    data/raw_datagouv/datagouv_candidats_results.parquet

Output:
- data/raw_datagouv/datagouv_candidats_results.parquet

Run from project root:
    python src/get_data/download/download_datagouv_candidats_results.py

Force re-download:
    python src/get_data/download/download_datagouv_candidats_results.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATAGOUV_DIR = PROJECT_ROOT / "data" / "raw_datagouv"
OUTPUT_PATH = RAW_DATAGOUV_DIR / "datagouv_candidats_results.parquet"

DATASET_API_URL = "https://www.data.gouv.fr/api/1/datasets/6481e741d4cf002ec0efec9d/"
CANDIDATS_PARQUET_RESOURCE_ID = "4d3b35f6-0b22-4415-a24c-419a676312e2"

REQUIRED_COLUMNS = {
    "id_election",
    "code_departement",
    "code_commune",
    "code_bv",
    "voix",
    "ratio_voix_inscrits",
    "ratio_voix_exprimes",
    "nom",
    "prenom",
}

TARGET_ELECTIONS = {
    "2008_muni_t1",
    "2008_muni_t2",
    "2014_muni_t1",
    "2014_muni_t2",
    "2020_muni_t1",
    "2020_muni_t2",
}


def build_request(url: str) -> urllib.request.Request:
    """Build a request accepted by data.gouv.fr and object storage."""
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "validation-pipeline-votes/1.0",
        },
    )


def read_json_url(url: str) -> dict:
    """Read JSON from a remote URL with clear errors."""
    try:
        with urllib.request.urlopen(build_request(url), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach data.gouv.fr dataset API: {url}\n{exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dataset API did not return valid JSON: {url}") from exc


def find_candidats_parquet_url(dataset_metadata: dict) -> str:
    """
    Resolve the current download URL from the data.gouv.fr dataset metadata.

    Primary match:
    - stable resource id for candidats_results.parquet.

    Fallback:
    - any parquet resource whose title contains 'candidat'.
    """
    resources = dataset_metadata.get("resources", [])

    for resource in resources:
        if resource.get("id") == CANDIDATS_PARQUET_RESOURCE_ID:
            url = resource.get("url")
            if not url:
                raise RuntimeError(
                    "The candidate parquet resource exists, but has no download URL."
                )
            return url

    for resource in resources:
        title = str(resource.get("title", "")).lower()
        fmt = str(resource.get("format", "")).lower()
        if "candidat" in title and fmt == "parquet":
            url = resource.get("url")
            if url:
                return url

    available = [
        {
            "id": resource.get("id"),
            "title": resource.get("title"),
            "format": resource.get("format"),
        }
        for resource in resources
    ]

    raise RuntimeError(
        "Could not find the data.gouv.fr candidate parquet resource.\n"
        f"Expected resource id: {CANDIDATS_PARQUET_RESOURCE_ID}\n"
        f"Available resources: {available}"
    )


def download_file(url: str, output_path: Path) -> None:
    """
    Download to a temporary file first, then move into place.

    This prevents leaving a half-written parquet file if the download fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(build_request(url), timeout=300) as response:
            with tmp_path.open("wb") as fh:
                shutil.copyfileobj(response, fh)

        validate_parquet(tmp_path)
        tmp_path.replace(output_path)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def validate_parquet(path: Path) -> None:
    """
    Validate schema and confirm that the Paris municipal rows needed downstream
    are present.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)

    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise RuntimeError(
            f"Downloaded parquet is missing required columns: {missing}\n"
            f"Path: {path}"
        )

    check_df = pd.read_parquet(path, columns=["code_departement", "id_election"])
    paris_municipal_rows = check_df[
        (check_df["code_departement"].astype(str) == "75")
        & (check_df["id_election"].astype(str).isin(TARGET_ELECTIONS))
    ]

    if paris_municipal_rows.empty:
        raise RuntimeError(
            "Downloaded parquet is valid, but contains no Paris municipal rows "
            "for the expected municipal elections: "
            f"{sorted(TARGET_ELECTIONS)}"
        )

    print(f"[ok] Valid parquet: {path.relative_to(PROJECT_ROOT)}")
    print(f"[ok] Paris municipal candidate rows found: {len(paris_municipal_rows):,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download data.gouv.fr candidate-level election results parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the parquet already exists and validates.",
    )
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        print(f"Found existing file: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
        validate_parquet(OUTPUT_PATH)
        print("Use --force to re-download it.")
        return

    print("Resolving current data.gouv.fr candidate parquet URL...")
    dataset_metadata = read_json_url(DATASET_API_URL)
    download_url = find_candidats_parquet_url(dataset_metadata)

    print(f"Downloading candidate results parquet to {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ...")
    download_file(download_url, OUTPUT_PATH)

    validate_parquet(OUTPUT_PATH)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[ok] Download complete: {size_mb:.1f} MB")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
