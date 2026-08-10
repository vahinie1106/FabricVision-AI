# Reliable Windows backend start for FabricVision-AI.
# Frees a leftover uvicorn on :8000 (WinError 10013), then starts reload server.
# Keeps port 8000 so the Next.js client (http://127.0.0.1:8000) stays valid.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:PYTHONPATH = (Get-Location).Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (Test-Path ".\venv\Scripts\python.exe") {
    $py = ".\venv\Scripts\python.exe"
} else {
    $py = "python"
}

& $py -m backend_api.utils.port_check 8000
if ($LASTEXITCODE -ne 0) {
    Write-Error "Port 8000 is not available for FabricVision-AI."
    exit $LASTEXITCODE
}

Write-Host "Starting: uvicorn backend_api.main:app --reload --host 127.0.0.1 --port 8000"
& $py -m uvicorn backend_api.main:app --reload --host 127.0.0.1 --port 8000
