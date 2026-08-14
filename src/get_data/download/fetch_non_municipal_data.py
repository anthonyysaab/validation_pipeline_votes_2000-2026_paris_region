"""Download the expected non-municipal Paris election datasets."""

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "download_manifest.csv"
BASE_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
HTTP_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024

EXPECTED_DATASETS = (
    "elections-europeennes-2009",
    "elections-europeennes-2014",
    "elections-europeennes-2019",
    "elections-europeennes-2024",
    "elections-legislatives-2025-1ertour",
    "elections-presidentielles-2007-1ertour",
    "elections-presidentielles-2007-2emetour",
    "elections-presidentielles-2012-1ertour",
    "elections-presidentielles-2012-2emetour",
    "elections-presidentielles-2017-1ertour",
    "elections-presidentielles-2017-2emetour",
    "elections-presidentielles-2022-2emetour",
    "elections-regionales-2010-1ertour",
    "elections-regionales-2010-2emetour",
    "elections-regionales-2015-1ertour",
    "elections-regionales-2015-2emetour",
    "elections-regionales-2021-1ertour",
    "elections-regionales-2021-2emetour",
)


def validate_frame(dataset_id: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"{dataset_id} returned no rows")
    columns = {str(column).strip().lower() for column in frame.columns}
    required_groups = ({"scrutin", "type_election"}, {"annee"}, {"tour", "numero_tour"})
    missing = ["/".join(sorted(group)) for group in required_groups if not columns & group]
    if missing:
        raise ValueError(f"{dataset_id} is missing required columns: {missing}")


def download_atomically(dataset_id: str, url: str, destination: Path) -> pd.DataFrame:
    temporary = destination.with_suffix(".parquet.download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "paris-election-pipeline/1.0"})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            declared_size = int(response.headers.get("Content-Length", "0"))
            if declared_size > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(f"{dataset_id} exceeds the {MAX_DOWNLOAD_BYTES}-byte limit")

            downloaded = 0
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(
                            f"{dataset_id} exceeded the {MAX_DOWNLOAD_BYTES}-byte limit"
                        )
                    output.write(chunk)

        frame = pd.read_parquet(temporary)
        validate_frame(dataset_id, frame)
        temporary.replace(destination)
        return frame
    finally:
        temporary.unlink(missing_ok=True)


def get_datasets(force_download: bool = False) -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    datasets = {}
    manifest = []

    for dataset_id in EXPECTED_DATASETS:
        local_path = DATA_DIR / f"{dataset_id}.parquet"
        url = f"{BASE_URL}/{dataset_id}/exports/parquet"

        if force_download or not local_path.is_file():
            print(f"downloading {dataset_id}...")
            frame = download_atomically(dataset_id, url, local_path)
            source = "downloaded"
        else:
            print(f"loading cached {dataset_id}...")
            frame = pd.read_parquet(local_path)
            validate_frame(dataset_id, frame)
            source = "cache"

        datasets[dataset_id] = frame
        manifest.append(
            {
                "dataset_id": dataset_id,
                "status": "OK",
                "source": source,
                "rows": len(frame),
                "columns": len(frame.columns),
                "path": str(local_path.relative_to(PROJECT_ROOT)),
            }
        )
        print(f"{dataset_id}: {frame.shape[0]} rows, {frame.shape[1]} columns")

    if len(datasets) != len(EXPECTED_DATASETS):
        raise RuntimeError(
            f"Loaded {len(datasets)} of {len(EXPECTED_DATASETS)} required datasets"
        )

    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    print(f"[ok] Download manifest written to: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download cached datasets")
    args = parser.parse_args()
    get_datasets(force_download=args.force)


if __name__ == "__main__":
    main()
