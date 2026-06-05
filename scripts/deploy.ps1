#!/usr/bin/env pwsh
# Blue-green deployment switcher for maize-leaf-classifier (Windows / PowerShell).
# Usage: .\scripts\deploy.ps1 [-Tag <git-sha>]
# Requirements: Docker Desktop with Linux containers, PowerShell 5.1+
param(
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$SlotFile    = Join-Path $ProjectRoot ".active-slot"
$ConfFile    = Join-Path $ProjectRoot "docker\conf.d\active-upstream.conf"
$ComposeFile = "docker/docker-compose.yml"

$MaxRetries    = 10
$RetryInterval = 5

# Determine active and inactive slots
$CurrentSlot = "blue"
if (Test-Path $SlotFile) {
    $CurrentSlot = (Get-Content $SlotFile -Raw).Trim()
}
$InactiveSlot = if ($CurrentSlot -eq "blue") { "green" } else { "blue" }

Write-Host "[deploy] Current: $CurrentSlot -> Target: $InactiveSlot (tag: $Tag)"

Push-Location $ProjectRoot
try {
    $env:API_TAG = $Tag

    # Pull new image — ignore failure for local-only builds
    docker compose -f $ComposeFile pull "api-$InactiveSlot" 2>&1 | Out-Null

    # Restart inactive slot with the new image
    docker compose -f $ComposeFile up -d --no-deps --force-recreate "api-$InactiveSlot"
    if ($LASTEXITCODE -ne 0) { throw "[deploy] Failed to start api-$InactiveSlot" }

    # Poll health of inactive slot directly (bypassing nginx)
    $Healthy = $false
    $HealthScript = 'import urllib.request,json,sys;r=urllib.request.urlopen("http://localhost:8000/health",timeout=8);d=json.loads(r.read());print(d.get("status","unknown"))'

    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Host "[deploy] Health check $i/$MaxRetries..."
        $Status = docker compose -f $ComposeFile exec -T "api-$InactiveSlot" python -c $HealthScript 2>$null
        if ($Status -eq "ok") {
            $Healthy = $true
            Write-Host "[deploy] api-$InactiveSlot is healthy!"
            break
        }
        Write-Host "[deploy] Not ready yet ($Status). Waiting ${RetryInterval}s..."
        Start-Sleep -Seconds $RetryInterval
    }

    if (-not $Healthy) {
        Write-Error "[deploy] FAILED: api-$InactiveSlot did not become healthy after $MaxRetries attempts.`nTraffic remains on api-$CurrentSlot. No changes made to nginx."
        exit 1
    }

    # Rewrite upstream config and do a graceful nginx reload (zero-downtime)
    $Timestamp   = (Get-Date -AsUTC -Format "yyyy-MM-ddTHH:mm:ssZ")
    $ConfContent = @"
# Managed by deploy.ps1 -- last updated: $Timestamp -- slot: $InactiveSlot
upstream active_api {
    server api-${InactiveSlot}:8000;
}
"@
    # Write with Unix line endings so nginx (Linux) is happy
    [System.IO.File]::WriteAllText($ConfFile, $ConfContent.Replace("`r`n", "`n"))

    docker compose -f $ComposeFile exec -T nginx nginx -s reload
    if ($LASTEXITCODE -ne 0) { throw "[deploy] nginx reload failed" }

    # Persist the new active slot
    [System.IO.File]::WriteAllText($SlotFile, $InactiveSlot)

    Write-Host "[deploy] Done. Active slot: $InactiveSlot (tag: $Tag)."
    Write-Host "[deploy] Previous slot (api-$CurrentSlot) still running -- run rollback.ps1 to revert."
} finally {
    Pop-Location
}
