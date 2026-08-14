# Paris Election Data Validation Pipeline

Python pipeline for downloading, cleaning, validating, and auditing Paris election results. It covers European, legislative, presidential, regional, and municipal files available within the 2000-2026 project window.

## Outputs

- `data/clean/election_results_long.parquet`: candidate-level results
- `data/clean/election_results_wide_vote_share.parquet`: polling-station vote shares
- `data/validation/`: row-count, vote-total, schema, and range checks
- `data/audit/`: raw-to-clean case correspondence checks
- `data/manifest/`: download and cleaning manifests

The checked-in clean dataset contains 220,404 candidate-level rows across 24 election files. Two source vote-total anomalies are recorded in `data/validation/expressed_vote_mismatches.csv`.

## Setup

Requires Python 3.11 and PowerShell on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Run

```powershell
.\run_pipeline.ps1
```

Each stage exits non-zero on download, cleaning, validation, or audit failure. A successful run writes fresh manifests and prints `[ok] Full pipeline completed.`

Run the local checks without downloading data:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m src.get_data.validation.validate_clean_election_results
```

## Data sources

- [Paris Open Data](https://opendata.paris.fr/)
- [data.gouv.fr election results](https://www.data.gouv.fr/)

Raw downloads are excluded from Git. Clean outputs, validation results, and manifests are versioned so results can be inspected without rerunning network stages.
