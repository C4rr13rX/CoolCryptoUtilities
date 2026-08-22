# R3V3N!R Control Tower launcher -- idempotent full-stack bring-up.
#
# Brings up (or leaves alone) every service the trading stack needs:
#   1. Brain substrate    (w1z4rd_node.exe on :8090)
#   2. R3V3N!R web panel  (Daphne ASGI + WebSockets on :8000)
#   3. Production manager (main.py --action start_production)
#   4. Brain feeder       (scripts/run_brain_feeder.py)  -- skipped if a
#                          history supervisor is currently training
#   5. Market evolution   (included W1z4rDV1510n protected-fold GA)
#
# Each check is by listening-port (brain, waitress) or by command-line
# substring (prod_manager, brain_feeder). Already-running services are
# left alone. Opens the dashboard URL at the end either way.
#
# Designed for the desktop shortcut -- double-click recovers the stack
# regardless of what's currently up.

$projectRoot   = "D:\Projects\CoolCryptoUtilities"
$python        = "$projectRoot\.venv\Scripts\python.exe"
$brainBin      = "D:\Projects\W1z4rDV1510n\bin\w1z4rd_node.exe"
$brainProject  = "D:\Projects\W1z4rDV1510n"
$evolutionPython = "C:\Python313\python.exe"
$evolutionScript = "$brainProject\scripts\market_evolution_service.py"
$evolutionWatchdog = "$brainProject\scripts\market_evolution_watchdog.py"
$evolutionState  = "$brainProject\runtime\market-evolution"
$brainDataDir  = "D:\w1z4rdv1510n-data"
$webRoot       = "$projectRoot\web"
$logsDir       = "$projectRoot\logs"
$panelHost     = "127.0.0.1"
$panelPort     = 8001
$brainPort     = 8090
$threads       = 8

# Wallet identity -- public address, not a secret. Workaround for
# default_env_user returning None outside the manage.py boot path,
# which can leave PortfolioState unable to derive the wallet.
$primaryWallet = "0x291c854811e92906a658fb94aa511bf919f968ad"

# -- helpers ---------------------------------------------------------------

function Test-Port($port) {
    $conn = $null
    try {
        $conn = New-Object System.Net.Sockets.TcpClient
        $pending = $conn.BeginConnect($panelHost, $port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(750)) { return $false }
        $conn.EndConnect($pending)
        return $conn.Connected
    } catch { return $false }
    finally { if ($conn) { $conn.Dispose() } }
}

function Find-PythonProcess($needle) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$needle*" }
}

function Find-Process($name) {
    Get-Process -Name $name -ErrorAction SilentlyContinue
}

function Wait-Port($port, $name, $maxSeconds = 60) {
    $i = 0
    while ($i -lt ($maxSeconds * 2)) {
        Start-Sleep -Milliseconds 500
        if (Test-Port $port) {
            Write-Host "  $name ready on :$port"
            return $true
        }
        $i++
    }
    Write-Host "  WARN: $name did not come up on :$port after $maxSeconds s"
    return $false
}

# -- 1. Brain substrate ----------------------------------------------------
#
# Non-blocking by design: nothing in the trading path gates on the brain
# (it's a supplemental confidence signal), so we NEVER hold up the panel or
# production waiting for it. The node is launched via start_node.ps1 -- the
# single source of truth for its args (api --addr) and env (identity pools,
# RAM politeness floor, auto-checkpoint). Launching the bare binary the old
# way started a process that never bound :8090, so the previous 300 s
# Wait-Port always timed out -- that was the "gets stuck" hang.

Write-Host "[1/5] Brain substrate"
$brainStarter = "$brainProject\start_node.ps1"
$brainProc = Find-Process "w1z4rd_node"
if ($brainProc) {
    $rssGb = [math]::Round($brainProc.WorkingSet64 / 1GB, 2)
    Write-Host "  already running -- pid=$($brainProc.Id) RSS=${rssGb}GB"
} elseif (Test-Path $brainStarter) {
    Write-Host "  starting via start_node.ps1 (background, non-blocking)..."
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-ExecutionPolicy","Bypass","-WindowStyle","Hidden","-File",$brainStarter `
        -WorkingDirectory $brainProject `
        -WindowStyle Hidden
    # Courtesy probe only -- do NOT block the stack on it. If it's slow to
    # bind, trading proceeds anyway and the brain joins when ready.
    if (Wait-Port $brainPort "brain" 15) {
        Write-Host "  brain online on :$brainPort"
    } else {
        Write-Host "  brain still coming up -- continuing without waiting (trading does not depend on it)"
    }
} else {
    Write-Host "  WARN: start_node.ps1 not found at $brainStarter (skipping brain)"
}

# -- 2. R3V3N!R web panel --------------------------------------------------

Write-Host "[2/5] R3V3N!R web panel"
if (Test-Port $panelPort) {
    Write-Host "  already running on :$panelPort"
} else {
    Write-Host "  starting Waitress Django server..."
    $env:WAITRESS_HOST    = $panelHost
    $env:WAITRESS_PORT    = "$panelPort"
    $env:WAITRESS_THREADS = "$threads"
    Start-Process -FilePath $python `
        -ArgumentList "run_waitress.py" `
        -WorkingDirectory $webRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$logsDir\web_waitress_8001.log" `
        -RedirectStandardError  "$logsDir\web_waitress_8001.err"
    Wait-Port $panelPort "panel" 30 | Out-Null
}

# -- 3. Production manager (trading bot) -----------------------------------

Write-Host "[3/5] Production manager"
$prodProc = Find-PythonProcess "start_production"
if ($prodProc) {
    Write-Host "  already running -- count=$($prodProc.Count)"
} else {
    Write-Host "  starting..."
    $env:PRIMARY_WALLET       = $primaryWallet
    $env:SECURE_ENV_HYDRATED  = ""   # force re-hydration from vault
    # Don't force SKIP_TF_CONFIGURE -- let pipeline._load_tf attempt the
    # import once, log a single clear WARNING if it can't load, then
    # cache the failure permanently in-process. Other systems (model_lab
    # GA, brain_regime) that depend on TF then either GET TF or see a
    # visible failure they can act on -- rather than being silently
    # disabled by an opinionated default.
    $env:SKIP_TF_CONFIGURE    = $null
    Start-Process -FilePath $python `
        -ArgumentList "-u","main.py","--action","start_production","--stay-alive" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$logsDir\prod_direct.log" `
        -RedirectStandardError  "$logsDir\prod_direct.err"
    Start-Sleep -Seconds 5
    $prodProc = Find-PythonProcess "start_production"
    if ($prodProc) {
        Write-Host "  spawned -- count=$($prodProc.Count)"
    } else {
        Write-Host "  WARN: production manager did not appear in process list"
    }
}

# -- 4. Brain feeder (skipped while a supervisor is training) --------------

Write-Host "[4/5] Brain feeder"
$supervisorRunning = Find-PythonProcess "brain_history_supervisor"
if ($supervisorRunning) {
    Write-Host "  history supervisor is training -- skipping feeder to avoid lock contention"
} else {
    $feederProc = Find-PythonProcess "run_brain_feeder"
    if ($feederProc) {
        Write-Host "  already running -- count=$($feederProc.Count)"
    } else {
        Write-Host "  starting..."
        Start-Process -FilePath $python `
            -ArgumentList "scripts/run_brain_feeder.py" `
            -WorkingDirectory $projectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput "$logsDir\feeder_direct.log" `
            -RedirectStandardError  "$logsDir\feeder_direct.err"
    }
}

# -- 5. Protected market-brain evolution ---------------------------------

Write-Host "[5/5] Protected market-brain evolution"
$evolutionProc = Find-PythonProcess "market_evolution_watchdog.py"
if ($evolutionProc) {
    Write-Host "  supervisor already running -- count=$($evolutionProc.Count)"
} elseif ((Test-Path $evolutionScript) -and (Test-Path $evolutionWatchdog)) {
    $stopMarker = Join-Path $evolutionState "STOP"
    if (Test-Path -LiteralPath $stopMarker) {
        Remove-Item -LiteralPath $stopMarker -Force
        Write-Host "  cleared cooperative STOP marker"
    }
    New-Item -ItemType Directory -Force $evolutionState | Out-Null
    Write-Host "  starting persistent RAM-aware supervisor..."
    Start-Process -FilePath $evolutionPython `
        -ArgumentList "-u",$evolutionWatchdog,"--python",$evolutionPython,"--service",$evolutionScript,"--state-dir",$evolutionState,"--min-free-memory-gb","3.5","--memory-poll-seconds","15","--restart-delay-seconds","30","--","--population","8","--workers","1","--brain-gate-every","1","--test-days","28" `
        -WorkingDirectory $brainProject `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$evolutionState\supervisor.stdout.log" `
        -RedirectStandardError  "$evolutionState\supervisor.stderr.log"
    Start-Sleep -Seconds 2
    $evolutionProc = Find-PythonProcess "market_evolution_watchdog.py"
    if ($evolutionProc) {
        Write-Host "  supervisor spawned -- pid=$($evolutionProc.ProcessId)"
        Write-Host "  accuracy improvements: $evolutionState\accuracy_improvements.jsonl"
    } else {
        Write-Host "  WARN: market evolution supervisor did not appear in process list"
    }
} else {
    Write-Host "  WARN: evolution service or supervisor script is missing"
}

# -- open the panel --------------------------------------------------------

Write-Host ""
Write-Host "Opening dashboard..."
Start-Process "http://${panelHost}:${panelPort}/"
