param([switch]$Refresh, [int]$Port = 4747)
$ErrorActionPreference = 'Stop'
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Open http://127.0.0.1:$Port or stop the existing GUI before refreshing."
}
Set-Location $PSScriptRoot
$metadataPath = Join-Path $PSScriptRoot '.gitnexus/gitnexus.json'
if (-not (Test-Path -LiteralPath $metadataPath)) {
    throw 'GitNexus index missing. Install GitNexus and run: gitnexus analyze --index-only'
}
$metadataJson = & node -e 'const m=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8")); console.log(JSON.stringify({runnerIdentity:m.runnerIdentity}));' $metadataPath
if ($LASTEXITCODE -ne 0) { throw 'Could not read GitNexus metadata with Node.js.' }
$metadata = $metadataJson | ConvertFrom-Json
$node = $metadata.runnerIdentity.runtime.executablePath
$cli = $metadata.runnerIdentity.invokedArtifact.path
if (-not (Test-Path -LiteralPath $node) -or -not (Test-Path -LiteralPath $cli)) {
    throw 'The indexed GitNexus installation is unavailable. Reinstall GitNexus and refresh the index.'
}
if ($Refresh) {
    & $node $cli analyze --force --index-only --workers 2
    if ($LASTEXITCODE -ne 0) { throw 'GitNexus indexing failed.' }
}
Write-Host "GitNexus learning GUI: http://127.0.0.1:$Port"
Write-Host 'Keep this terminal running. Press Ctrl+C to stop. Use -Refresh after source changes.'
& $node $cli serve --host 127.0.0.1 --port $Port
