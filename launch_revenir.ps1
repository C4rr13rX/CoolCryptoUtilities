# R3V3N!R Control Tower launcher -- idempotent full-stack bring-up.
#
# Brings up (or leaves alone) every service the trading stack needs:
#   1. Brain substrate    (w1z4rd_node.exe on :8090)
#   2. R3V3N!R web panel  (Daphne ASGI + WebSockets on :8000)
#   3. Production manager (main.py --action start_production)
#   4. Brain feeder       (scripts/run_brain_feeder.py)  -- skipped if a
#                          history supervisor is currently training
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
$brainDataDir  = "D:\w1z4rdv1510n-data"
$webRoot       = "$projectRoot\web"
$logsDir       = "$projectRoot\logs"
$panelHost     = "127.0.0.1"
$panelPort     = 8000
$brainPort     = 8090
$threads       = 8

# Wallet identity -- public address, not a secret. Workaround for
# default_env_user returning None outside the manage.py boot path,
# which can leave PortfolioState unable to derive the wallet.
$primaryWallet = "0x291c854811e92906a658fb94aa511bf919f968ad"

# -- helpers ---------------------------------------------------------------

function Test-Port($port) {
    try {
        $conn = New-Object System.Net.Sockets.TcpClient
        $conn.Connect($panelHost, $port) | Out-Null
        $conn.Close()
        return $true
    } catch { return $false }
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

Write-Host "[1/4] Brain substrate"
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

# -- 2. R3V3N!R web panel (ASGI) -------------------------------------------

Write-Host "[2/4] R3V3N!R web panel"
if (Test-Port $panelPort) {
    Write-Host "  already running on :$panelPort"
} else {
    Write-Host "  starting Daphne ASGI server..."
    $env:WAITRESS_HOST    = $panelHost
    $env:WAITRESS_PORT    = "$panelPort"
    $env:WAITRESS_THREADS = "$threads"
    Start-Process -FilePath $python `
        -ArgumentList "run_asgi.py" `
        -WorkingDirectory $webRoot `
        -WindowStyle Minimized
    Wait-Port $panelPort "panel" 30 | Out-Null
}

# -- 3. Production manager (trading bot) -----------------------------------

Write-Host "[3/4] Production manager"
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

Write-Host "[4/4] Brain feeder"
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

# -- open the panel --------------------------------------------------------

Write-Host ""
Write-Host "Opening dashboard..."
Start-Process "http://${panelHost}:${panelPort}/"
