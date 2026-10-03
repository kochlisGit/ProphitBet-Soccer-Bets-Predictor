# Project-local Dots installation. Supply OPENROUTER_API_KEY in your environment.
# Extra CLI options can be passed through, e.g. .\start-dots.ps1 --help
$ErrorActionPreference = 'Stop'
$dotsArguments = @($args)
$dotsModel = 'openrouter/free'
for ($i = 0; $i -lt $dotsArguments.Count; $i++) {
    if ($dotsArguments[$i] -eq '--model') {
        if ($i + 1 -ge $dotsArguments.Count) { throw '--model requires a free model ID.' }
        $dotsModel = [string]$dotsArguments[++$i]
    } elseif ($dotsArguments[$i] -like '--model=*') {
        $dotsModel = $dotsArguments[$i].Substring(8)
    } else {
        continue
    }
    if ($dotsModel -ne 'openrouter/free' -and
        ($dotsModel -notmatch '^[^/]+/[^:]+:free$' -or $dotsModel.StartsWith('openrouter/'))) {
        throw 'Paid/automatic models are disabled by this launcher. Use openrouter/free or an explicit provider/model:free.'
    }
}
$dotsRoot = Join-Path $PSScriptRoot 'tools/dots'
$dotsPython = Join-Path $dotsRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $dotsPython)) {
    throw 'Dots is not installed in tools/dots/.venv.'
}
$previousDotsHome = $env:INVISIBLE_MCP_HOME
$env:INVISIBLE_MCP_HOME = Join-Path $dotsRoot '.runtime'
Push-Location $dotsRoot
try {
    # Explicit CLI default overrides any paid model in the environment or .env.
    # OpenRouter's free router only routes to free models; no paid fallback.
    Write-Host "Dots model: $dotsModel (free tier; provider quotas still apply)"
    & $dotsPython -m dots --model openrouter/free @dotsArguments
    $dotsExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    $env:INVISIBLE_MCP_HOME = $previousDotsHome
}
exit $dotsExitCode
