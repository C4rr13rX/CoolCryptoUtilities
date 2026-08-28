<#
.SYNOPSIS
    Drives Claude Code until R3V3N!R is LIVE TRADING and PROFITABLE.

.DESCRIPTION
    The in-session cron could only fire while the assistant was idle, so it
    never ran during continuous work and died with the session. This does the
    job from OUTSIDE: an independent PowerShell loop that owns the schedule,
    re-invokes Claude on the SAME session, and refuses to stop until a real
    live trade has been placed and shows a profit.

    Each pass:
      1. reads the ten-link path check for ground truth,
      2. stops when live trades exist AND live P/L is positive,
      3. otherwise re-invokes Claude with the first failing link,
      4. VERIFIES Claude actually responded -- a silent or errored run is
         retried, never counted as progress,
      5. restarts production if it has died or gone quiet.

.NOTES
    Everything is echoed to this window and appended to
    data\GetToLiveTrading.log. Newest output is always at the bottom.
#>

[CmdletBinding()]
param(
    [string] $Repo        = "D:\Projects\CoolCryptoUtilities",
    # EMPTY means "start a fresh session each pass", which is the right
    # default and the reason this parameter changed.
    #
    # It used to hard-code bbff78d9, an eight-day session that had grown to
    # 33MB / ~844k tokens of context. --resume replays that entire history on
    # EVERY pass, so a 10-minute loop re-read ~844k tokens six times an hour
    # and hit the session limit before any fix could land. The session had
    # already compacted three times and re-inflated within a day each time;
    # it cannot shrink back.
    #
    # Each pass is self-contained anyway -- the prompt carries the path-check
    # output and the loop reads ground truth from the DB, so there is nothing
    # in the conversation history the next pass needs. Pass an explicit id
    # only to deliberately continue one specific session.
    [string] $SessionId   = "",
    # How long to let the system run between nudges.
    [int]    $IntervalSec = 600,
    # How long a single Claude invocation may take before it is abandoned.
    [int]    $ClaudeTimeoutSec = 1500,
    [switch] $Once,

    # Local Django dashboard. These are dev defaults for a localhost-bound
    # site (the project already defaults ADMIN_EMAIL/ADMIN_PASSWORD to
    # admin/admin in serverless/hybrid/migrate_to_s3.py), passed to Claude so
    # it can check the UI without stopping to ask. Override here if the local
    # site ever uses anything else -- and if this site is ever exposed beyond
    # localhost, change the password and stop passing it in a prompt.
    [string] $DashboardPort = "8001",
    [string] $DashboardUser = "admin",
    [string] $DashboardPass = "admin"
)

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "R3V3N!R -> LIVE TRADING"

# SINGLE INSTANCE ONLY.
#
# Five copies of this script ran at once on 2026-08-27, each invoking
# `claude --resume` on the SAME session independently. That consumed 97% of a
# session limit in one afternoon. A mutex is held for the life of the process
# and released automatically if it is killed, so a second copy exits instead of
# doubling the spend.
$script:Mutex = New-Object System.Threading.Mutex($false, "Global\R3V3N1R_GetToLiveTrading")
if (-not $script:Mutex.WaitOne(0)) {
    Write-Host "Another GetToLiveTrading loop is already running. Exiting." -ForegroundColor Red
    exit 1
}

# NEVER LEAVE AN ORPHANED CLAUDE BEHIND.
#
# Start-Process children survive their parent on Windows. Killing this loop
# window therefore left a claude process running with nothing reading its
# output: it kept working, and kept spending tokens, until it finished or
# timed out. Observed 2026-08-28 during a restart, and the user had to shut
# it down by hand.
#
# So tear the child down on any exit path -- Ctrl-C, Stop-Process, or a
# normal break out of the loop.
$script:ActiveClaude = $null

function Stop-ActiveClaude {
    if ($script:ActiveClaude -and -not $script:ActiveClaude.HasExited) {
        try {
            Write-Host "  stopping in-flight Claude (PID $($script:ActiveClaude.Id))..." -ForegroundColor Yellow
            Stop-Process -Id $script:ActiveClaude.Id -Force -ErrorAction SilentlyContinue
        } catch { }
    }
    $script:ActiveClaude = $null
}

# Fires on normal exit and on Ctrl-C. Note this does NOT fire on
# `Stop-Process -Force`, which is why the startup sweep below also exists --
# belt and braces, because the failure mode is invisible token spend.
try {
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Stop-ActiveClaude } | Out-Null
} catch { }

# SWEEP ORPHANS FROM A PREVIOUS RUN.
#
# If a previous loop was force-killed, its claude child is still out there
# working against a prompt nobody will read. We hold the single-instance
# mutex by this point, so any claude process older than this one belongs to
# a dead loop and is safe to reap.
$swept = 0
try {
    Get-CimInstance Win32_Process -Filter "Name='node.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*claude*' -and $_.CommandLine -notlike '*GetToLiveTrading*' } |
        ForEach-Object {
            $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.ParentProcessId)" -ErrorAction SilentlyContinue
            # Reap only claude processes whose parent is gone -- a live parent
            # means someone else (an interactive session) owns it.
            if (-not $owner) {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                $swept++
            }
        }
} catch { }
if ($swept -gt 0) {
    Write-Host "swept $swept orphaned claude process(es) from a previous run" -ForegroundColor Yellow
}

# Keep the newest line in view. Without this the console keeps the viewport
# where the user last left it, so a long-running loop appears frozen while it
# is actually scrolling far below.
function Scroll-ToBottom {
    try {
        $ui = $Host.UI.RawUI
        $pos = $ui.CursorPosition
        $win = $ui.WindowPosition
        $height = $ui.WindowSize.Height
        if ($pos.Y -ge ($win.Y + $height - 1)) {
            $win.Y = [Math]::Max(0, $pos.Y - $height + 2)
            $ui.WindowPosition = $win
        }
    } catch { }
}

$Python  = Join-Path $Repo ".venv\Scripts\python.exe"
$LogFile = Join-Path $Repo "data\GetToLiveTrading.log"

# Start-Process cannot launch the bare `claude` shim -- it is a shell script,
# and Windows answers "%1 is not a valid Win32 application". claude.cmd is the
# real entry point.
$ClaudeExe = (Get-Command claude.cmd -ErrorAction SilentlyContinue).Source
if (-not $ClaudeExe) { $ClaudeExe = "C:\Users\Adam\AppData\Roaming\npm\claude.cmd" }

# ---------------------------------------------------------------- logging --

function Write-Line {
    param(
        [string] $Message,
        [string] $Colour = "Gray"
    )
    $stamp = (Get-Date).ToString("HH:mm:ss")
    $line  = "[$stamp] $Message"
    Write-Host $line -ForegroundColor $Colour
    Scroll-ToBottom
    try { Add-Content -Path $LogFile -Value $line -Encoding UTF8 } catch { }
}

function Write-Banner {
    param([string] $Text, [string] $Colour = "Cyan")
    $bar = "=" * 78
    Write-Host ""
    Write-Host $bar -ForegroundColor $Colour
    Write-Host "  $Text" -ForegroundColor $Colour
    Write-Host $bar -ForegroundColor $Colour
    try {
        Add-Content -Path $LogFile -Value "`n$bar`n  $Text`n$bar" -Encoding UTF8
    } catch { }
}

# ------------------------------------------------------------ ground truth --

function Get-TradingState {
    <#  The single source of truth. Reads the database directly rather than
        trusting any report file -- a stale report is how "all gates pass"
        coexisted with zero live trades for hours. #>
    $script = @'
import json, sqlite3, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = {}
try:
    c = sqlite3.connect("file:storage/trading_cache.db?mode=ro", uri=True)
    now = time.time()
    q = lambda s, *a: list(c.execute(s, a))[0][0]
    out["live_rows"]   = q("SELECT COUNT(*) FROM trading_ops WHERE status LIKE 'live%'")
    out["ticks_10m"]   = q("SELECT COUNT(*) FROM market_stream WHERE ts > ?", now - 600)
    out["ghost_1h"]    = q("SELECT COUNT(*) FROM trading_ops WHERE status='ghost-entry' AND ts > ?", now - 3600)
    out["cycles_10m"]  = q("SELECT COUNT(*) FROM organism_snapshots WHERE ts > ?", now - 600)
    out["transitions"] = q("SELECT COUNT(*) FROM feedback_events WHERE source='live_transition'")
    row = list(c.execute("SELECT usd_amount FROM balances WHERE wallet='guardian' AND chain='base' AND symbol='USDC'"))
    out["usdc"] = round(float(row[0][0]), 4) if row else None
except Exception as exc:
    out["db_error"] = "%s: %s" % (type(exc).__name__, exc)

# Live P/L comes from the lifetime registry, which survives ledger resets.
out["live_pl"] = None
out["live_trades"] = 0
try:
    sys.path.insert(0, ".")
    from services import strategy_registry
    total, trades = 0.0, 0
    for row in strategy_registry.list_strategies():
        live = ((row.get("lifetime") or {}).get("live")) or {}
        total  += float(live.get("total_profit") or 0.0)
        trades += int(live.get("trades") or 0)
    out["live_pl"] = round(total, 6)
    out["live_trades"] = trades
except Exception:
    pass
print(json.dumps(out))
'@
    $tmp = Join-Path $env:TEMP "revenir_state.py"
    Set-Content -Path $tmp -Value $script -Encoding UTF8
    try {
        $raw = & $Python $tmp 2>$null
        if ($raw) { return ($raw | ConvertFrom-Json) }
    } catch { }
    return $null
}

function Get-PathCheck {
    try {
        $out = & $Python (Join-Path $Repo "scripts\live_path_check.py") 2>&1 | Out-String
        return $out
    } catch {
        return "path check failed: $_"
    }
}

function Get-FirstFailure {
    param([string] $Report)
    foreach ($line in ($Report -split "`n")) {
        if ($line -match "\[FAIL") { return $line.Trim() }
    }
    return $null
}

# ------------------------------------------------------------- production --

function Get-ProductionCount {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -like '*start_production*' }
    if ($null -eq $procs) { return 0 }
    return @($procs).Count
}

function Restart-Production {
    Write-Line "restarting production..." "Yellow"
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*start_production*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 6
    $log = Join-Path $Repo "data\production.log"
    Start-Process -FilePath $Python `
                  -ArgumentList "-u","main.py","--action","start_production","--stay-alive" `
                  -WorkingDirectory $Repo -WindowStyle Hidden `
                  -RedirectStandardOutput $log -RedirectStandardError (Join-Path $Repo "data\production.err.log")
    Write-Line "production relaunched" "Yellow"
}

# ----------------------------------------------------------------- claude --

function Invoke-Claude {
    <#  Re-invoke Claude on the SAME session and VERIFY it responded.
        A silent or failed run is reported as such rather than being counted
        as a completed pass -- that verification is the whole point. #>
    param([string] $Prompt)

    if ([string]::IsNullOrWhiteSpace($SessionId)) {
        Write-Line "invoking Claude (fresh session) ..." "Magenta"
    } else {
        Write-Line "invoking Claude on session $SessionId ..." "Magenta"
    }
    $started  = Get-Date
    $outFile  = Join-Path $env:TEMP "claude_out_$([guid]::NewGuid().ToString('N')).txt"
    $errFile  = "$outFile.err"
    $promptFile = Join-Path $env:TEMP "claude_prompt_$([guid]::NewGuid().ToString('N')).txt"
    Set-Content -Path $promptFile -Value $Prompt -Encoding UTF8

    try {
        # Fresh session unless one was explicitly requested. Resuming is the
        # expensive path: it re-reads the whole prior conversation every pass.
        $claudeArgs = if ([string]::IsNullOrWhiteSpace($SessionId)) {
            @("-p", "--permission-mode", "bypassPermissions")
        } else {
            @("--resume", $SessionId, "-p", "--permission-mode", "bypassPermissions")
        }
        # Remember the child so a shutdown can take it with us. Killing the
        # loop window used to leave the claude process running with nothing
        # reading its output -- it kept working, and kept spending tokens,
        # invisibly. Observed 2026-08-28 during a restart.
        $proc = Start-Process -FilePath $ClaudeExe `
            -ArgumentList $claudeArgs `
            -WorkingDirectory $Repo -NoNewWindow -PassThru `
            -RedirectStandardInput $promptFile `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $script:ActiveClaude = $proc

        # Show the work AS IT HAPPENS.
        #
        # WaitForExit blocked here silently for up to $ClaudeTimeoutSec (25
        # minutes by default). Claude's output goes to a temp file that
        # nothing read until the process exited, so a pass that was landing
        # commits looked identical to a hung one -- observed 2026-08-28,
        # where a 17-minute pass produced three commits while the window sat
        # apparently frozen on "invoking Claude".
        #
        # So poll instead: stream new output as it is written, and print a
        # heartbeat carrying elapsed time, commits landed this pass, and live
        # pipeline counters. A watched window should never have to guess
        # whether anything is happening.
        $headSha    = (& git -C $Repo rev-parse --short HEAD 2>$null)
        $lastLen    = 0
        $lastBeat   = Get-Date
        $beatEvery  = 20      # seconds between heartbeats
        $deadline   = (Get-Date).AddSeconds($ClaudeTimeoutSec)
        $timedOut   = $false

        while (-not $proc.HasExited) {
            if ((Get-Date) -gt $deadline) { $timedOut = $true; break }
            Start-Sleep -Milliseconds 700

            # --- stream whatever Claude has written since last look ---
            try {
                if (Test-Path $outFile) {
                    $now = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
                    if ($null -ne $now -and $now.Length -gt $lastLen) {
                        $chunk = $now.Substring($lastLen)
                        $lastLen = $now.Length
                        foreach ($ln in ($chunk -split "`r?`n")) {
                            if ($ln.Trim()) {
                                Write-Host "  | $($ln.TrimEnd())" -ForegroundColor Gray
                            }
                        }
                        Scroll-ToBottom
                        $lastBeat = Get-Date
                    }
                }
            } catch { }

            # --- heartbeat when it has been quiet ---
            if (((Get-Date) - $lastBeat).TotalSeconds -ge $beatEvery) {
                $lastBeat = Get-Date
                $secs = [int]((Get-Date) - $started).TotalSeconds
                $left = [int]($deadline - (Get-Date)).TotalSeconds

                $newCommits = ""
                try {
                    $head = (& git -C $Repo rev-parse --short HEAD 2>$null)
                    if ($head -and $headSha -and $head -ne $headSha) {
                        $n = (& git -C $Repo rev-list --count "$headSha..HEAD" 2>$null)
                        $subject = (& git -C $Repo log -1 --format=%s 2>$null)
                        $newCommits = "  committed $n -> $subject"
                    }
                } catch { }

                $live = ""
                try {
                    $st = Get-TradingState
                    if ($st) {
                        $live = ("  ticks10m={0} ghost1h={1} live={2}" -f `
                                 $st.ticks_10m, $st.ghost_1h, $st.live_rows)
                    }
                } catch { }

                Write-Line ("  ...working {0}s (timeout in {1}s){2}{3}" -f `
                            $secs, $left, $live, $newCommits) "DarkCyan"
            }
        }

        if ($timedOut) {
            Write-Line "Claude exceeded ${ClaudeTimeoutSec}s; abandoning this pass" "Red"
            try { $proc.Kill() } catch { }
            return "failed"
        }
    } catch {
        Write-Line "could not start Claude: $_" "Red"
        return "failed"
    }

    $script:ActiveClaude = $null
    $elapsed = [int]((Get-Date) - $started).TotalSeconds
    $reply   = ""
    if (Test-Path $outFile) { $reply = (Get-Content $outFile -Raw -ErrorAction SilentlyContinue) }

    # VERIFICATION: a response must exist and carry real content. An empty
    # reply, or one that is only a quota/limit notice, is NOT a response.
    if ([string]::IsNullOrWhiteSpace($reply)) {
        $err = ""
        if (Test-Path $errFile) { $err = (Get-Content $errFile -Raw -ErrorAction SilentlyContinue) }
        Write-Line "Claude produced NO output after ${elapsed}s -- will retry" "Red"
        if ($err) { Write-Line "  stderr: $($err.Substring(0, [Math]::Min(300, $err.Length)))" "DarkRed" }
        return "failed"
    }
    if ($reply -match "(?i)(usage limit|rate limit|resets at|resets? [0-9]|quota exceeded)" -and $reply.Length -lt 400) {
        Write-Line "SESSION LIMIT REACHED -- Claude returned a limit notice, not work" "Red"
        Write-Line "  $($reply.Trim())" "DarkRed"
        $script:LimitNotice = $reply.Trim()
        return "limit"
    }

    Write-Line "Claude responded after ${elapsed}s ($($reply.Length) chars)" "Green"
    Write-Host ""
    Write-Host "----- Claude -------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host $reply.Trim()
    Write-Host "--------------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
    try { Add-Content -Path $LogFile -Value $reply -Encoding UTF8 } catch { }

    Remove-Item $outFile, $errFile, $promptFile -ErrorAction SilentlyContinue
    return "ok"
}

# --------------------------------------------------------------- quota --

function Get-ResetWait {
    <#  Work out how long to wait for the session quota to return.

        Claude's limit notice carries the reset time ("resets 2:40am"), so
        parse it and sleep until then rather than hammering the API on a
        fixed retry. An unparseable notice falls back to a conservative wait.

        Returns seconds to wait.  #>
    param([string] $Notice)

    $fallback = 1800

    if ([string]::IsNullOrWhiteSpace($Notice)) { return $fallback }

    # "resets 2:40am", "resets at 14:05", "resets 2am"
    if ($Notice -match "(?i)resets?(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?") {
        $hh = [int]$Matches[1]
        $mm = if ($Matches[2]) { [int]$Matches[2] } else { 0 }
        $ap = $Matches[3]
        if ($ap) {
            $ap = $ap.ToLower()
            if ($ap -eq "pm" -and $hh -lt 12) { $hh += 12 }
            if ($ap -eq "am" -and $hh -eq 12) { $hh = 0 }
        }
        try {
            $now    = Get-Date
            $target = Get-Date -Hour $hh -Minute $mm -Second 0
            # A reset time already past today means it lands tomorrow.
            if ($target -le $now) { $target = $target.AddDays(1) }
            $wait = [int]($target - $now).TotalSeconds
            # Add a small cushion so we do not race the reset itself.
            $wait += 60
            if ($wait -gt 0 -and $wait -lt 86400) { return $wait }
        } catch { }
    }

    return $fallback
}

function Wait-ForQuota {
    <#  Sleep until the quota is back, saying so out loud.

        The loop must survive running out of session time: it waits for the
        window to reopen and then resumes on its own, rather than dying or
        spinning. Progress is printed so a watched window never looks hung.  #>
    param([int] $Seconds)

    $resumeAt = (Get-Date).AddSeconds($Seconds)
    Write-Banner "OUT OF SESSION TIME -- WAITING FOR QUOTA" "Yellow"
    Write-Line ("will retry at {0} ({1} minutes from now)" -f `
                $resumeAt.ToString("HH:mm:ss"), [int]($Seconds / 60)) "Yellow"

    $remaining = $Seconds
    while ($remaining -gt 0) {
        $chunk = [Math]::Min(300, $remaining)
        Start-Sleep -Seconds $chunk
        $remaining -= $chunk
        if ($remaining -gt 0) {
            Write-Line ("  ...quota returns in {0} min (at {1})" -f `
                        [int]($remaining / 60), $resumeAt.ToString("HH:mm:ss")) "DarkGray"
        }
    }
    Write-Banner "QUOTA WINDOW REOPENED -- RESUMING" "Green"
    Write-Line ("resumed at {0}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")) "Green"
}

# ------------------------------------------------------------------- main --

Write-Banner "R3V3N!R  ->  LIVE PROFITABLE TRADING" "Cyan"
Write-Line "repo      : $Repo"
Write-Line ("session   : " + $(if ([string]::IsNullOrWhiteSpace($SessionId)) { "fresh each pass (no --resume)" } else { $SessionId }))
Write-Line "interval  : ${IntervalSec}s between passes"
Write-Line "log       : $LogFile"
Write-Line "dashboard : http://localhost:$DashboardPort  (login $DashboardUser/$DashboardPass, local dev only)"
Write-Line "Stops ONLY when live trades exist AND live P/L is positive." "White"

$pass = 0
while ($true) {
    $pass++
    Write-Banner "PASS $pass  --  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "Cyan"

    $state = Get-TradingState
    if ($null -eq $state) {
        Write-Line "could not read trading state" "Red"
    } else {
        $plText = if ($null -eq $state.live_pl) { "--" } else { "{0:+0.0000;-0.0000;0.0000}" -f $state.live_pl }
        Write-Line ("live_rows={0}  live_trades={1}  live_PL={2}" -f `
                    $state.live_rows, $state.live_trades, $plText) "White"
        Write-Line ("ticks10m={0}  ghost1h={1}  cycles10m={2}  transitions={3}  usdc={4}" -f `
                    $state.ticks_10m, $state.ghost_1h, $state.cycles_10m, $state.transitions, $state.usdc)
    }

    # ---- the only exit condition ----
    if ($state -and $state.live_rows -ge 1 -and $state.live_trades -ge 1 -and
        $null -ne $state.live_pl -and $state.live_pl -gt 0) {
        Write-Banner "GOAL REACHED: LIVE TRADING AND PROFITABLE" "Green"
        Write-Line ("live trades = {0}   live P/L = {1}" -f $state.live_trades, $state.live_pl) "Green"
        break
    }
    if ($state -and $state.live_rows -ge 1) {
        Write-Line "LIVE TRADES EXIST but P/L is not positive yet -- continuing" "Yellow"
    }

    # ---- keep production alive ----
    $procs = Get-ProductionCount
    Write-Line "production processes: $procs"
    if ($procs -eq 0) {
        Write-Line "production is DOWN" "Red"
        Restart-Production
    } elseif ($state -and $state.cycles_10m -eq 0 -and $state.ticks_10m -eq 0) {
        Write-Line "production alive but idle for 10 minutes" "Red"
        Restart-Production
    }

    # ---- what is actually blocking ----
    Write-Line "running the ten-link path check..."
    $report  = Get-PathCheck
    foreach ($line in ($report -split "`n")) {
        if ($line -match "\[(PASS|FAIL| )") { Write-Host "    $($line.TrimEnd())" }
    }
    $failure = Get-FirstFailure -Report $report
    if ($failure) { Write-Line "first failing link: $failure" "Yellow" }
    else          { Write-Line "every link passes; waiting on the executor" "Green" }

    # ---- nudge Claude, and verify it answered ----
    $prompt = @"
AUTONOMOUS LIVE-TRADING LOOP -- pass $pass. Do not stop until R3V3N!R has
placed a REAL live trade and live P/L is positive.

Current ground truth from the database:
  live_rows   = $($state.live_rows)
  live_trades = $($state.live_trades)
  live_P/L    = $($state.live_pl)
  ticks_10m   = $($state.ticks_10m)
  cycles_10m  = $($state.cycles_10m)
  transitions = $($state.transitions)
  usdc        = $($state.usdc)

First failing link: $failure

Work ONLY on that link. Diagnose it with real measurements -- query the DB,
read the code, check on-chain. Never assume; verify. If a number looks too
good, it is probably an artifact: check it against an independent source
before believing it.

Then: fix the root cause, add a test that pins the behaviour, run the
relevant tests, commit, and restart production if needed.

Report concisely: which link failed, the evidence, the fix, and the next link.
If a live trade LOSES, verify the demotion guards fired and report the P/L
honestly -- never hide a loss.

Wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad on base.
Keep ENABLE_GAS_REFILL=0 -- the refill drained the wallet three times.

Dashboard, if you need to check it or log in:
  URL      http://localhost:$DashboardPort
  admin    http://localhost:$DashboardPort/admin/
  login    $DashboardUser / $DashboardPass
These are LOCAL DEV credentials for a site bound to localhost. Do not send
them anywhere off this machine, and do not put them in a commit.
The project reads the same pair from ADMIN_EMAIL / ADMIN_PASSWORD
(see serverless/hybrid/migrate_to_s3.py).
"@

    # Compare with the STRING on the left.
    #
    # PowerShell's -eq coerces the right operand to the LEFT operand's type,
    # so `$true -eq "limit"` is TRUE -- any non-empty string casts to $true.
    # Invoke-Claude used to return $true on success, so every SUCCESSFUL pass
    # was read as a quota limit and slept 30 minutes for nothing. Observed
    # 2026-08-28 10:13: a completed 3580-char report with commits landed was
    # followed by "OUT OF SESSION TIME".
    #
    # Invoke-Claude now returns "ok" / "limit" / "failed", and every test puts
    # the literal first so no boolean coercion can happen again.
    $answered = Invoke-Claude -Prompt $prompt
    if ("limit" -eq $answered) {
        # Out of session time. Wait for the window to reopen and try the same
        # pass again -- do not count it as progress, and do not give up.
        Wait-ForQuota -Seconds (Get-ResetWait -Notice $script:LimitNotice)
        $pass--
        continue
    }
    if ("ok" -ne $answered) {
        Write-Line "no verified response; retrying in 120s" "Red"
        Start-Sleep -Seconds 120
        continue
    }

    if ($Once) { Write-Line "single pass requested; exiting" "Cyan"; break }

    # Count down out loud. A silent sleep is indistinguishable from a hang,
    # and this window is meant to be watched.
    Write-Line "letting the system run for ${IntervalSec}s before the next pass" "DarkGray"
    $remaining = $IntervalSec
    while ($remaining -gt 0) {
        $chunk = [Math]::Min(30, $remaining)
        Start-Sleep -Seconds $chunk
        $remaining -= $chunk
        if ($remaining -gt 0) {
            $s = Get-TradingState
            if ($s) {
                Write-Line ("  ...{0}s left  live={1} ticks10m={2} cycles10m={3} usdc={4}" -f `
                            $remaining, $s.live_rows, $s.ticks_10m, $s.cycles_10m, $s.usdc) "DarkGray"
                if ($s.live_rows -ge 1) {
                    Write-Line "LIVE TRADE APPEARED -- breaking out early" "Green"
                    break
                }
            } else {
                Write-Line "  ...${remaining}s left" "DarkGray"
            }
        }
    }
}

Write-Banner "LOOP FINISHED" "Green"
Write-Line "Full transcript: $LogFile" "Green"
