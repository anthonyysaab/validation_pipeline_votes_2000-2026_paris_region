PYTHON ?= ./.venv/Scripts/python.exe

.PHONY: all get_data/download clean validate audit pipeline test

all: pipeline

get_data/download:
	$(PYTHON) -m src.get_data.download.fetch_non_municipal_data
	$(PYTHON) -m src.get_data.download.download_datagouv_candidats_results
	$(PYTHON) -m src.get_data.download.extract_municipales_datagouv

clean:
	$(PYTHON) -m src.get_data.clean.clean_election_results
	$(PYTHON) -m src.get_data.clean.clean_municipal_results
	$(PYTHON) -m src.get_data.clean.build_clean_results

validate:
	$(PYTHON) -m src.get_data.validation.validate_clean_election_results

audit:
	$(PYTHON) -m src.get_data.audit.audit_raw_to_clean_cases

pipeline: get_data/download clean validate audit

test:
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy
