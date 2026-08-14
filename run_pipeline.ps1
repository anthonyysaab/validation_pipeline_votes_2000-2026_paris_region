$ErrorActionPreference = "Stop"

$PY = ".\.venv\Scripts\python.exe"

function Invoke-PipelineStep {
    param([string]$Label, [string]$Module)

    Write-Host "`n$Label"
    & $PY -m $Module
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline step failed with exit code $LASTEXITCODE`: $Module"
    }
}

Invoke-PipelineStep "[1/8] Downloading non-municipal OpenData Paris files..." "src.get_data.download.fetch_non_municipal_data"
Invoke-PipelineStep "[2/8] Downloading data.gouv.fr candidate results..." "src.get_data.download.download_datagouv_candidats_results"
Invoke-PipelineStep "[3/8] Extracting municipal data.gouv.fr files..." "src.get_data.download.extract_municipales_datagouv"
Invoke-PipelineStep "[4/8] Cleaning non-municipal results..." "src.get_data.clean.clean_election_results"
Invoke-PipelineStep "[5/8] Cleaning municipal results..." "src.get_data.clean.clean_municipal_results"
Invoke-PipelineStep "[6/8] Building final clean results..." "src.get_data.clean.build_clean_results"
Invoke-PipelineStep "[7/8] Validating clean results..." "src.get_data.validation.validate_clean_election_results"
Invoke-PipelineStep "[8/8] Auditing raw-to-clean cases..." "src.get_data.audit.audit_raw_to_clean_cases"

Write-Host "`n[ok] Full pipeline completed."
