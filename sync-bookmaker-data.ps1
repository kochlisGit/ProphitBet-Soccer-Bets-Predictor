[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$collectorPython = Join-Path $PSScriptRoot 'tools/dots/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $collectorPython)) {
    throw 'Install Dots and its Python environment first.'
}
& $collectorPython (Join-Path $PSScriptRoot 'tools/collect_bookmaker_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Browser collection failed; import cancelled.' }
Push-Location (Join-Path $PSScriptRoot 'prophitbet-saas')
try {
    docker compose exec -T backend python -m backend.app.services.bookmaker_feed
    if ($LASTEXITCODE -ne 0) { throw 'Bookmaker import failed.' }
} finally {
    Pop-Location
}
