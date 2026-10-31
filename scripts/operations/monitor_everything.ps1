param(
    [switch]$Once,
    [int]$RefreshSeconds = 10
)

$ErrorActionPreference = "Continue"
$CryptoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WizardRoot = "D:\Projects\W1z4rDV1510n"
$Heartbeat = Join-Path $CryptoRoot "logs\production_manager_heartbeat.json"
$MarketStatus = Join-Path $WizardRoot "runtime\market-evolution\status.json"
$Champion = Join-Path $WizardRoot "runtime\market-evolution\champion.json"
$SeniorState = Join-Path $WizardRoot "runtime\programming-brain-codex-watch\state.json"

function Test-LoopbackPort([int]$Port) {
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object {
            $_.LocalAddress -in @("127.0.0.1", "::1", "0.0.0.0", "::") -and
            ($_.State -eq "Listen" -or ($_.RemotePort -eq 0 -and $_.OwningProcess -gt 0))
        } | Select-Object -First 1
    if ($connection) { return "UP pid=$($connection.OwningProcess)" }
    return "DOWN"
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json } catch { return $null }
}

do {
    if (-not $Once) { Clear-Host }
    Write-Host "WIZARD VISION - COMPLETE OPERATIONS STATUS" -ForegroundColor Cyan
    Write-Host ("Updated {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ""
    Write-Host ("Django dashboard       : {0}" -f (Test-LoopbackPort 8001))
    Write-Host ("Crypto brain node      : {0}" -f (Test-LoopbackPort 8090))
    Write-Host ("Senior brain proxy     : {0}" -f (Test-LoopbackPort 18096))

    try {
        $ef = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/edge-search/status/" -TimeoutSec 12 -ErrorAction Stop
        $wCount = if ($ef.workers) { @($ef.workers).Count } else { 0 }
        Write-Host ("EdgeForge GA workers   : {0} active (agent={1}; profitable_rules={2}; skipped_no_profit={3})" -f `
            $wCount, $ef.config.research_agent, $ef.counts.profitable_findings, $ef.counts.complete_no_profit_skipped)
    } catch {
        Write-Host "EdgeForge GA workers   : status unavailable" -ForegroundColor Yellow
    }

    $taProc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*tradingagent_worker*" } | Select-Object -First 1
    $evProc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*evolve_strategies*" } | Select-Object -First 1
    Write-Host ("TradingAgent (AGY)     : {0}" -f $(if ($taProc) { "UP pid=$($taProc.ProcessId)" } else { "DOWN" }))
    Write-Host ("Strategy evolution     : {0}" -f $(if ($evProc) { "UP pid=$($evProc.ProcessId)" } else { "DOWN" }))

    $heartbeat = Read-Json $Heartbeat
    if ($heartbeat) {
        Write-Host ("Production manager     : {0}; live_ready={1}; precision={2}; samples={3}" -f `
            $heartbeat.status, $heartbeat.meta.live_ready, $heartbeat.meta.live_precision, $heartbeat.meta.live_samples)
        Write-Host ("Ghost/live safety      : ghost active; live promotion remains gated")
    } else {
        Write-Host "Production manager     : heartbeat unavailable" -ForegroundColor Yellow
    }

    $market = Read-Json $MarketStatus
    $champion = Read-Json $Champion
    if ($market -and $champion) {
        $summary = $champion.result.summary
        Write-Host ("Market evolution       : {0}; generation={1}; RAM free={2} GB" -f `
            $market.phase, $market.generation, $market.available_memory_gb)
        Write-Host ("Champion OOS           : accuracy={0:N4}; coverage={1:N4}; PF={2:N4}; expectancy={3:N6}" -f `
            $summary.min_accuracy, $summary.min_coverage, $summary.min_profit_factor, $summary.min_expectancy)
    } else {
        Write-Host "Market evolution       : status unavailable" -ForegroundColor Yellow
    }

    $senior = Read-Json $SeniorState
    if ($senior) {
        $probe = $senior.last_probe
        Write-Host ("Senior brain training  : {0}/{1}; supervisor={2}; wrapper={3}; worker={4}" -f `
            $probe.status.phase, $probe.status.state, $probe.supervisor_count, $probe.wrapper_count, $probe.worker_count)
        Write-Host ("Curriculum             : {0:N0} accepted; {1:N0} quarantined; {2:N0} forward remain" -f `
            $probe.curriculum.accepted_rows, $probe.curriculum.deferred_rows, $probe.curriculum.forward_remaining_rows)
    } else {
        Write-Host "Senior brain training  : watcher snapshot unavailable" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Dashboard: http://127.0.0.1:8001/" -ForegroundColor Green
    if (-not $Once) {
        Write-Host ("Refreshing every {0}s. Ctrl+C closes only this monitor." -f $RefreshSeconds)
        Start-Sleep -Seconds ([Math]::Max(2, $RefreshSeconds))
    }
} while (-not $Once)
