#!/usr/bin/env pwsh
# Immediate rollback to previous slot — no health check (assumes previous slot is still running).
# Usage: .\scripts\rollback.ps1
$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$SlotFile    = Join-Path $ProjectRoot ".active-slot"
$ConfFile    = Join-Path $ProjectRoot "docker\conf.d\active-upstream.conf"
$ComposeFile = "docker/docker-compose.yml"

if (-not (Test-Path $SlotFile)) {
    Write-Error "[rollback] ERROR: .active-slot file not found. Cannot determine current slot."
    exit 1
}

$CurrentSlot = (Get-Content $SlotFile -Raw).Trim()
$PrevSlot    = if ($CurrentSlot -eq "blue") { "green" } else { "blue" }

Write-Host "[rollback] Rolling back: $CurrentSlot -> $PrevSlot"

$Timestamp   = (Get-Date -AsUTC -Format "yyyy-MM-ddTHH:mm:ssZ")
$ConfContent = @"
# Managed by rollback.ps1 -- last updated: $Timestamp -- slot: $PrevSlot (ROLLBACK)
upstream active_api {
    server api-${PrevSlot}:8000;
}
"@
[System.IO.File]::WriteAllText($ConfFile, $ConfContent.Replace("`r`n", "`n"))

Push-Location $ProjectRoot
try {
    docker compose -f $ComposeFile exec -T nginx nginx -s reload
    if ($LASTEXITCODE -ne 0) { throw "[rollback] nginx reload failed" }
} finally {
    Pop-Location
}

[System.IO.File]::WriteAllText($SlotFile, $PrevSlot)
Write-Host "[rollback] Done. Active slot is now: $PrevSlot"
