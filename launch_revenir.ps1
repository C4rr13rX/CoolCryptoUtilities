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
# A PROCESS IS NOT A SERVICE. The check used to be "does a process named
# w1z4rd_node exist", which a process that started and never bound :8090
# passes forever -- so the launcher saw "already running", skipped it, and
# the brain stayed unreachable indefinitely. Observed 2026-09-05: PID 1276
# alive since 08:38 with nothing listening on 8090 and /health actively
# refusing the connection, across several launcher runs that each reported
# it healthy.
#
# The port is what callers actually need, so the port is what is checked. A
# process that is up but not serving is treated as down and restarted --
# after being stopped, because two of them would fight over the port.
$brainProc = Find-Process "w1z4rd_node"
$brainListening = $null -ne (Get-NetTCPConnection -LocalPort $brainPort -State Listen -ErrorAction SilentlyContinue)
if ($brainProc -and -not $brainListening) {
    Write-Host "  pid=$($brainProc.Id) is running but NOT listening on :$brainPort -- restarting it"
    Stop-Process -Id $brainProc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $brainProc = $null
}
if ($brainProc -and $brainListening) {
    $rssGb = [math]::Round($brainProc.WorkingSet64 / 1GB, 2)
    Write-Host "  already running -- pid=$($brainProc.Id) RSS=${rssGb}GB, serving :$brainPort"
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
    # -X utf8: see Restart-Production in GetToLiveTrading.ps1. Startup UTF-8
    # mode is what makes a bare open(path,"w") in library code UTF-8;
    # ensure_utf8_mode() can only fix this process's own stdout.
    Start-Process -FilePath $python `
        -ArgumentList "-X","utf8","-u","main.py","--action","start_production","--stay-alive" `
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

# -- 6. trading agent worker ----------------------------------------------
#
# The reading agent that hunts trades. Governed by its own start_at_boot
# setting so it can be left off deliberately rather than by accident: the
# launcher asks the database, and only skips when the answer is an explicit
# no. An unreadable setting starts it -- a pipeline that is silently not
# trading is the failure this whole stack exists to avoid.

Write-Host ""
Write-Host "Checking trading agent worker..."
$agentProc = Find-PythonProcess "tradingagent_worker"
if ($agentProc) {
    Write-Host "  already running -- pid=$($agentProc.ProcessId)"
} else {
    $startAgent = $true
    try {
        $answer = & $python "$webRoot\manage.py" shell -c "from tradingagent.models import AgentConfig; c=AgentConfig.load(); print('YES' if getattr(c,'start_at_boot',True) else 'NO')" 2>$null
        if ($answer -match 'NO') { $startAgent = $false }
    } catch {
        Write-Host "  could not read start_at_boot; starting it (a dark agent is worse)"
    }
    if ($startAgent) {
        Start-Process -FilePath $python `
            -ArgumentList "-X","utf8","$webRoot\manage.py","tradingagent_worker" `
            -WorkingDirectory $webRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logsDir "agent_worker.log") `
            -RedirectStandardError  (Join-Path $logsDir "agent_worker.err.log")
        Start-Sleep -Seconds 2
        $agentProc = Find-PythonProcess "tradingagent_worker"
        if ($agentProc) { Write-Host "  started -- pid=$($agentProc.ProcessId)" }
        else { Write-Host "  WARN: agent worker did not appear in the process list" }
    } else {
        Write-Host "  start_at_boot is off -- left stopped, by setting"
    }
}

# -- 6b. strategy evolution ------------------------------------------------
#
# Searches for new trading rules and publishes what survives a holdout across
# multiple seeds. Hourly, because the book grows by a trade at a time and a
# rule that clears the bar on 156 round trips may not clear it on 300.

Write-Host ""
Write-Host "Checking strategy evolution..."
$evolveProc = Find-PythonProcess "evolve_strategies"
if ($evolveProc) {
    Write-Host "  already running -- pid=$($evolveProc.ProcessId)"
} else {
    Start-Process -FilePath $python `
        -ArgumentList "-X","utf8","$webRoot\manage.py","evolve_strategies","--loop" `
        -WorkingDirectory $webRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsDir "evolve.log") `
        -RedirectStandardError  (Join-Path $logsDir "evolve.err.log")
    Start-Sleep -Seconds 2
    Write-Host "  started"
}

# -- 7. continuous refinement loop + console ------------------------------
#
# The loop that keeps fixing the pipeline, and the window that shows what it
# is doing. The console is separate from the loop on purpose: closing the
# window must not stop the work.

Write-Host ""
Write-Host "Checking continuous refinement loop..."
$refineRoot = "D:\Projects\ContinuousRefinement"
if (Test-Path "$refineRoot\scripts\loop.py") {
    $loopProc = Find-PythonProcess "config.revenir.json"
    if ($loopProc) {
        Write-Host "  already running -- pid=$($loopProc.ProcessId)"
    } else {
        Start-Process -FilePath "python" `
            -ArgumentList "scripts\loop.py","--config","config.revenir.json" `
            -WorkingDirectory $refineRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput "$refineRoot\data\revenir-stdout.log" `
            -RedirectStandardError  "$refineRoot\data\revenir-stderr.log"
        Start-Sleep -Seconds 2
        Write-Host "  loop started"
    }

    # The console. pythonw so it owns a window rather than a console host.
    $consoleUp = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -like "*console.py*revenir*" }
    if ($consoleUp) {
        Write-Host "  console already open -- pid=$($consoleUp.ProcessId)"
    } else {
        Start-Process -FilePath "pythonw" `
            -ArgumentList "scripts\console.py","--config","config.revenir.json" `
            -WorkingDirectory $refineRoot
        Write-Host "  console opened"
    }
} else {
    Write-Host "  WARN: ContinuousRefinement is not installed at $refineRoot"
}

# -- open the panel --------------------------------------------------------

Write-Host ""
Write-Host "Opening dashboard..."
Start-Process "http://${panelHost}:${panelPort}/"
