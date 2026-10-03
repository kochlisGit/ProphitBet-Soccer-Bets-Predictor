#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run a SonarQube full-codebase scan for ProphitBet.

.DESCRIPTION
    Waits for SonarQube to be ready, generates an analysis token if SONAR_TOKEN
    is not already set, then launches sonar-scanner against the entire repo.

.EXAMPLE
    .\run-sonar-scan.ps1
    .\run-sonar-scan.ps1 -Token "squ_abc123"
#>

param(
    [string]$Token = $env:SONAR_TOKEN,
    [string]$AdminPassword = $(if ($env:SONAR_ADMIN_PASSWORD) { $env:SONAR_ADMIN_PASSWORD } else { "admin" }),
    [string]$SonarUrl = "http://localhost:9100",
    [string]$InternalUrl = "http://sonarqube:9000"
)

$ErrorActionPreference = "Stop"
$Compose = "docker compose"

# Read SONAR_TOKEN from prophitbet-saas/.env if not explicitly provided
if (-not $Token -and (Test-Path "$PSScriptRoot\prophitbet-saas\.env")) {
    $envLine = Get-Content "$PSScriptRoot\prophitbet-saas\.env" | Where-Object { $_ -match '^\s*SONAR_TOKEN\s*=\s*(.+)$' }
    if ($envLine) {
        $Token = $Matches[1].Trim()
    }
}

Write-Host "`n=== ProphitBet SonarQube Scanner ===" -ForegroundColor Cyan

# ── 1. Wait for SonarQube to be ready ──────────────────────────────────────────
Write-Host "`n[1/4] Waiting for SonarQube to be ready at $SonarUrl ..." -ForegroundColor Yellow
$maxWait = 120
$waited = 0
do {
    Start-Sleep -Seconds 5
    $waited += 5
    try {
        $status = (Invoke-RestMethod "$SonarUrl/api/system/status" -TimeoutSec 3).status
    } catch { $status = "UNREACHABLE" }
    Write-Host "      Status: $status ($waited s elapsed)"
} while ($status -ne "UP" -and $waited -lt $maxWait)

if ($status -ne "UP") {
    Write-Error "SonarQube did not become ready within $maxWait seconds. Is it running? `ndocker compose up -d sonarqube"
}
Write-Host "  SonarQube is UP." -ForegroundColor Green

# ── 2. Generate a token if none provided ───────────────────────────────────────
if (-not $Token) {
    Write-Host "`n[2/4] Generating analysis token (admin:***) ..." -ForegroundColor Yellow
    $cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:$AdminPassword"))
    $headers = @{ Authorization = "Basic $cred" }

    try {
        # Revoke any stale token first (ignore error if it doesn't exist)
        try {
            Invoke-RestMethod "$SonarUrl/api/user_tokens/revoke" -Method POST -Headers $headers `
                -Body "name=prophitbet-ci" | Out-Null
        } catch {}

        $resp = Invoke-RestMethod "$SonarUrl/api/user_tokens/generate" -Method POST -Headers $headers `
            -Body "name=prophitbet-ci&type=GLOBAL_ANALYSIS_TOKEN"
        $Token = $resp.token
        Write-Host "  Token generated. Add to .env as: SONAR_TOKEN=$Token" -ForegroundColor Green
        Write-Host "  (or pass: .\run-sonar-scan.ps1 -Token `"$Token`")" -ForegroundColor DarkGray
    } catch {
        Write-Host "`n[!] Authentication failed with SonarQube ($($_.Exception.Message))." -ForegroundColor Red
        Write-Host "    The admin password has likely been changed from the default ('admin')." -ForegroundColor Yellow
        Write-Host "`n    To resolve, you can either:" -ForegroundColor Yellow
        Write-Host "    1. Pass your admin password:  .\run-sonar-scan.ps1 -AdminPassword `"your_password`"" -ForegroundColor Cyan
        Write-Host "    2. Pass an analysis token:    .\run-sonar-scan.ps1 -Token `"squ_your_token`"" -ForegroundColor Cyan
        Write-Host "       (Create one at: $SonarUrl/account/security)" -ForegroundColor DarkGray
        Write-Host "    3. Reset the admin password back to 'admin' in PostgreSQL." -ForegroundColor Cyan
        throw
    }
} else {
    Write-Host "`n[2/4] Using provided SONAR_TOKEN." -ForegroundColor Yellow
}

# ── 3. Create the project in SonarQube if it doesn't exist ────────────────────
Write-Host "`n[3/4] Ensuring SonarQube project 'prophitbet' exists ..." -ForegroundColor Yellow
$authHeaders = @{ Authorization = "Bearer $Token" }
try {
    Invoke-RestMethod "$SonarUrl/api/projects/create" -Method POST -Headers $authHeaders `
        -Body "project=prophitbet&name=ProphitBet+Soccer+Predictor&mainBranch=main" | Out-Null
    Write-Host "  Project created." -ForegroundColor Green
} catch {
    Write-Host "  Project already exists (or creation skipped)." -ForegroundColor DarkGray
}

# ── 4. Run the scanner ────────────────────────────────────────────────────────
Write-Host "`n[4/4] Running sonar-scanner (this may take a few minutes) ..." -ForegroundColor Yellow
$env:SONAR_TOKEN = $Token

Push-Location "$PSScriptRoot\prophitbet-saas"
try {
    & docker compose `
        --profile scan `
        run --rm `
        -e "SONAR_TOKEN=$Token" `
        -e "SONAR_HOST_URL=$InternalUrl" `
        sonar-scanner
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Sonar scanner failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "`n=== Scan complete! ===" -ForegroundColor Green
Write-Host "Open your results: $SonarUrl/dashboard?id=prophitbet" -ForegroundColor Cyan
