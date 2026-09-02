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
    # How long a single Claude invocation may run before it is abandoned.
    #
    # 0 = NO LIMIT, and that is the default.
    #
    # This used to kill a pass at 1500s. A pass doing real work -- reading the
    # codebase, running tests, driving a swap on-chain -- routinely runs
    # longer than that, and killing it discards everything it had done and
    # every token it had spent, then starts the next pass from scratch on the
    # same problem. Observed 2026-09-02: a pass was killed at 1365s while
    # actively streaming events.
    #
    # A hung pass is handled by the streaming heartbeat instead: if it goes
    # quiet you can see that in the window and stop it yourself. Set a
    # positive number here only if you deliberately want a hard cap.
    [int]    $ClaudeTimeoutSec = 0,
    [switch] $Once,

    # Local Django dashboard. These are dev defaults for a localhost-bound
    # site (the project already defaults ADMIN_EMAIL/ADMIN_PASSWORD to
    # admin/admin in serverless/hybrid/migrate_to_s3.py), passed to Claude so
    # it can check the UI without stopping to ask. Override here if the local
    # site ever uses anything else -- and if this site is ever exposed beyond
    # localhost, change the password and stop passing it in a prompt.
    # ---- what counts as done ----
    #
    # "live_trades >= 1 AND live_pl > 0" used to end the run, so a SINGLE fill
    # closing a fraction of a cent up printed "GOAL REACHED". That proves the
    # plumbing works; it is not evidence of an edge.
    [int]    $MinLiveTrades  = 5,
    [double] $MinProfitFactor = 1.5,

    # ---- when to give up ----
    #
    # Nothing used to stop this on losses: a losing run just kept nudging
    # while the wallet ground down. These halt it and shout.
    [double] $MaxLossUsd     = 1.00,
    [double] $MinWalletUsd   = 5.50,

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

# --------------------------------------------------------------- render --

function Write-ClaudeEvent {
    <#  Turn one stream-json event into a readable line.

        Shows what Claude is DOING -- the command it ran, the file it edited,
        the text it wrote -- rather than a spinner. Anything unrecognised is
        ignored rather than dumped as raw JSON, which would bury the signal.  #>
    param([string] $Line)

    $ev = $null
    try { $ev = $Line | ConvertFrom-Json -ErrorAction Stop } catch { return }
    if (-not $ev) { return }

    switch ($ev.type) {
        "system" {
            if ($ev.subtype -eq "init") {
                Write-Host "  [start] session $($ev.session_id)" -ForegroundColor DarkGray
            }
        }
        "rate_limit_event" {
            $ri = $ev.rate_limit_info
            if ($ri) {
                $pct = [int](([double]$ri.utilization) * 100)
                $col = if ($pct -ge 95) { "Red" } elseif ($pct -ge 80) { "Yellow" } else { "DarkGray" }
                Write-Host ("  [quota] {0} at {1}% used" -f $ri.rateLimitType, $pct) -ForegroundColor $col
                $script:LastRateLimit = $ri
            }
        }
        "assistant" {
            foreach ($c in $ev.message.content) {
                switch ($c.type) {
                    "text" {
                        foreach ($t in ($c.text -split "`n")) {
                            if ($t.Trim()) { Write-Host "  $($t.TrimEnd())" -ForegroundColor White }
                        }
                    }
                    "tool_use" {
                        $d = ""
                        if ($c.input.command)       { $d = $c.input.command }
                        elseif ($c.input.file_path) { $d = $c.input.file_path }
                        elseif ($c.input.pattern)   { $d = $c.input.pattern }
                        elseif ($c.input.prompt)    { $d = $c.input.prompt }
                        if ($d.Length -gt 150) { $d = $d.Substring(0,150) + " ..." }
                        $d = $d -replace "`r?`n", " "
                        Write-Host ("  > {0}: {1}" -f $c.name, $d) -ForegroundColor Cyan
                    }
                }
            }
        }
        "user" {
            foreach ($c in $ev.message.content) {
                if ($c.type -ne "tool_result") { continue }
                $txt = ""
                if ($c.content -is [string]) { $txt = $c.content }
                elseif ($c.content) { $txt = ($c.content | ForEach-Object { $_.text }) -join " " }
                $txt = ($txt -replace "`r?`n", " ").Trim()
                if (-not $txt) { continue }
                if ($txt.Length -gt 200) { $txt = $txt.Substring(0,200) + " ..." }
                $col = if ($c.is_error) { "Red" } else { "DarkGray" }
                Write-Host "    $txt" -ForegroundColor $col
            }
        }
        "result" {
            if ($ev.result) { $script:FinalResult = [string]$ev.result }
            if ($null -ne $ev.total_cost_usd) {
                Write-Host ("  [done] {0:N4} USD, {1} turns, {2}s" -f `
                    $ev.total_cost_usd, $ev.num_turns, [int]($ev.duration_ms / 1000)) -ForegroundColor DarkGreen
            }
        }
    }
}

# ----------------------------------------------------------------- inbox --

function Read-Inbox {
    <#  Pick up anything the user wrote while a pass was running.

        There was no way to steer this loop without stopping it: you could
        watch it work but not tell it anything, so a correction meant killing
        the run and restarting. Now `data\agent_inbox.md` is read at the top
        of every pass, injected into that pass's prompt, and moved aside so
        the same note is never delivered twice.

        Write to it any time -- mid-pass is fine. It lands on the next pass.  #>

    $inbox = Join-Path $Repo "data\agent_inbox.md"
    if (-not (Test-Path $inbox)) { return "" }

    $text = ""
    try { $text = (Get-Content $inbox -Raw -ErrorAction Stop) } catch { return "" }
    if ([string]::IsNullOrWhiteSpace($text)) { return "" }

    # Archive rather than delete, so a note is never silently lost and there
    # is a record of what was asked and when.
    try {
        $archive = Join-Path $Repo "data\agent_inbox_archive.md"
        Add-Content -Path $archive -Encoding UTF8 -Value (
            "`n### delivered $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (pass $pass)`n" + $text)
        Remove-Item $inbox -Force -ErrorAction SilentlyContinue
    } catch { }

    Write-Banner "MESSAGE FROM USER -- delivering to this pass" "Magenta"
    foreach ($ln in ($text -split "`r?`n")) {
        if ($ln.Trim()) { Write-Line "  $($ln.TrimEnd())" "Magenta" }
    }
    return $text.Trim()
}

# ------------------------------------------------------------------- sms --

function Send-Milestone {
    <#  Text a milestone. Never let a notification failure stop trading:
        every path is swallowed, because a dropped text is worth far less
        than an interrupted run.  #>
    param([string] $Text)
    try {
        Write-Line "  texting: $Text" "Magenta"
        & $Python (Join-Path $Repo "scripts
otify_sms.py") $Text 2>&1 | ForEach-Object {
            Write-Host "    $_" -ForegroundColor DarkGray
        }
    } catch {
        Write-Line "  (text failed: $_)" "DarkYellow"
    }
}

# Milestones fire ONCE each. Without this the loop would text the same
# "first live trade" every pass for as long as the condition held true.
$script:Milestones = @{}

function Send-MilestoneOnce {
    param([string] $Key, [string] $Text)
    if ($script:Milestones.ContainsKey($Key)) { return }
    $script:Milestones[$Key] = $true
    Send-Milestone -Text $Text
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
    # Count SETTLED live trades, not attempts.
    #
    # "status LIKE 'live%'" also matches live-entry-blocked and
    # live-dry-run-entry. On 2026-09-01 that read live_rows=6 when all six
    # were blocked or dry-run and no real money had ever been spent -- the
    # loop's own display was overstating progress toward its goal.
    out["live_rows"] = q(
        "SELECT COUNT(*) FROM trading_ops WHERE status LIKE 'live%' "
        "AND status NOT LIKE '%blocked%' AND status NOT LIKE '%dry-run%'"
    )
    out["live_attempts"] = q("SELECT COUNT(*) FROM trading_ops WHERE status LIKE 'live%'")
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
    wins, losses, gross_win, gross_loss = 0, 0, 0.0, 0.0
    for row in strategy_registry.list_strategies():
        live = ((row.get("lifetime") or {}).get("live")) or {}
        total  += float(live.get("total_profit") or 0.0)
        trades += int(live.get("trades") or 0)
        wins   += int(live.get("wins") or 0)
        losses += int(live.get("losses") or 0)
        gross_win  += abs(float(live.get("gross_win") or 0.0))
        gross_loss += abs(float(live.get("gross_loss") or 0.0))
    out["live_pl"] = round(total, 6)
    out["live_trades"] = trades
    out["live_wins"] = wins
    out["live_losses"] = losses
    out["gross_win"] = round(gross_win, 6)
    out["gross_loss"] = round(gross_loss, 6)
    # Profit factor: gross wins over gross losses. No losses yet with a
    # positive book is treated as passing rather than dividing by zero.
    if gross_loss > 0:
        out["profit_factor"] = round(gross_win / gross_loss, 4)
    elif gross_win > 0:
        out["profit_factor"] = 999.0
    else:
        out["profit_factor"] = 0.0
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
    $script:FinalResult = ""
    $script:LastRateLimit = $null
    $outFile  = Join-Path $env:TEMP "claude_out_$([guid]::NewGuid().ToString('N')).txt"
    $errFile  = "$outFile.err"
    $promptFile = Join-Path $env:TEMP "claude_prompt_$([guid]::NewGuid().ToString('N')).txt"
    Set-Content -Path $promptFile -Value $Prompt -Encoding UTF8

    try {
        # Fresh session unless one was explicitly requested. Resuming is the
        # expensive path: it re-reads the whole prior conversation every pass.
        # stream-json emits an event per step -- every tool call, command and
        # result -- instead of one text blob at the end. That is what makes
        # the window show actual work rather than "...working 400s".
        $fmt = @("--output-format", "stream-json", "--verbose")
        $claudeArgs = if ([string]::IsNullOrWhiteSpace($SessionId)) {
            @("-p") + $fmt + @("--permission-mode", "bypassPermissions")
        } else {
            @("--resume", $SessionId, "-p") + $fmt + @("--permission-mode", "bypassPermissions")
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
        $eventsSeen = 0       # stream events since the last heartbeat
        # $ClaudeTimeoutSec = 0 means run to completion, however long it takes.
        $noLimit    = ($ClaudeTimeoutSec -le 0)
        $deadline   = if ($noLimit) { [DateTime]::MaxValue } else { (Get-Date).AddSeconds($ClaudeTimeoutSec) }
        $timedOut   = $false

        while (-not $proc.HasExited) {
            if (-not $noLimit -and (Get-Date) -gt $deadline) { $timedOut = $true; break }
            Start-Sleep -Milliseconds 700

            # --- stream whatever Claude has written since last look ---
            try {
                if (Test-Path $outFile) {
                    $now = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
                    if ($null -ne $now -and $now.Length -gt $lastLen) {
                        $chunk = $now.Substring($lastLen)
                        $lastLen = $now.Length
                        foreach ($ln in ($chunk -split "`r?`n")) {
                            if (-not $ln.Trim()) { continue }
                            Write-ClaudeEvent -Line $ln
                        }
                        Scroll-ToBottom
                        # Deliberately does NOT reset $lastBeat.
                        #
                        # It used to. Streamed output arrives continuously
                        # during a working pass, so every chunk pushed the
                        # heartbeat deadline forward and it never fired --
                        # and streamed lines go to the CONSOLE, not the log
                        # file. Result: an 8-minute hole in the log while the
                        # window showed steady work, which reads as hung.
                        # The heartbeat is the thing that reaches the log, so
                        # it has to fire on wall-clock time regardless.
                        $eventsSeen += 1
                    }
                }
            } catch { }

            # --- heartbeat when it has been quiet ---
            if (((Get-Date) - $lastBeat).TotalSeconds -ge $beatEvery) {
                $lastBeat = Get-Date
                $secs = [int]((Get-Date) - $started).TotalSeconds
                $leftText = if ($noLimit) { "no limit" } else {
                    "{0}s left" -f [int]($deadline - (Get-Date)).TotalSeconds }

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

                $act = if ($eventsSeen -gt 0) { "  +{0} events" -f $eventsSeen } else { "  (quiet)" }
                $eventsSeen = 0
                Write-Line ("  ...working {0}s ({1}){2}{3}{4}" -f `
                            $secs, $leftText, $act, $live, $newCommits) "DarkCyan"
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
    # With stream-json the file holds events, not prose. The final answer was
    # captured from the "result" event as it streamed past; fall back to a
    # last-pass parse if the stream ended before we rendered it.
    $reply = $script:FinalResult
    if ([string]::IsNullOrWhiteSpace($reply) -and (Test-Path $outFile)) {
        try {
            foreach ($ln in (Get-Content $outFile -ErrorAction SilentlyContinue)) {
                if (-not $ln.Trim()) { continue }
                $ev = $null
                try { $ev = $ln | ConvertFrom-Json -ErrorAction Stop } catch { continue }
                if ($ev.type -eq "result" -and $ev.result) { $reply = [string]$ev.result }
            }
        } catch { }
    }

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
Write-Line ("pass limit: " + $(if ($ClaudeTimeoutSec -le 0) { "none - a pass runs to completion" } else { "${ClaudeTimeoutSec}s" }))
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
        if ($state.live_attempts -gt $state.live_rows) {
            Write-Line ("live attempts={0} of which SETTLED={1} (rest blocked/dry-run)" -f `
                        $state.live_attempts, $state.live_rows) "DarkYellow"
        }
    }

    # ---- milestones worth a text ----
    if ($state) {
        if ($state.live_rows -ge 1) {
            Send-MilestoneOnce -Key "first_live" -Text (
                "R3V3N!R: FIRST LIVE TRADE placed. trades={0} P/L={1:+0.0000;-0.0000;0.0000} wallet=`${2}" -f `
                $state.live_trades, $state.live_pl, $state.usdc)
        }
        if ($state.live_trades -ge 1 -and $state.live_pl -gt 0) {
            Send-MilestoneOnce -Key "first_profit" -Text (
                "R3V3N!R: FIRST PROFITABLE live P/L {0:+0.0000} over {1} trade(s). PF {2}. Need {3} trades at PF {4} to finish." -f `
                $state.live_pl, $state.live_trades, $state.profit_factor, $MinLiveTrades, $MinProfitFactor)
        }
        # Halfway to the trade count, so a long run still reports in.
        $half = [Math]::Max(1, [int]($MinLiveTrades / 2))
        if ($state.live_trades -ge $half) {
            Send-MilestoneOnce -Key "half_trades" -Text (
                "R3V3N!R: {0}/{1} live trades. P/L {2:+0.0000;-0.0000;0.0000} PF {3} W/L {4}/{5}" -f `
                $state.live_trades, $MinLiveTrades, $state.live_pl, $state.profit_factor, `
                $state.live_wins, $state.live_losses)
        }
    }

    # ---- stop on SUSTAINED profit, not a single lucky fill ----
    if ($state -and $state.live_rows -ge 1 -and
        $state.live_trades -ge $MinLiveTrades -and
        $null -ne $state.live_pl -and $state.live_pl -gt 0 -and
        $null -ne $state.profit_factor -and $state.profit_factor -ge $MinProfitFactor) {
        Write-Banner "GOAL REACHED: CONSISTENTLY PROFITABLE LIVE TRADING" "Green"
        Write-Line ("live trades   = {0}  (>= {1} required)" -f $state.live_trades, $MinLiveTrades) "Green"
        Write-Line ("live P/L      = {0:+0.0000;-0.0000;0.0000}" -f $state.live_pl) "Green"
        Write-Line ("profit factor = {0}  (>= {1} required)" -f $state.profit_factor, $MinProfitFactor) "Green"
        Write-Line ("wins/losses   = {0}/{1}" -f $state.live_wins, $state.live_losses) "Green"
        Send-Milestone -Text (
            "R3V3N!R GOAL REACHED: {0} live trades, P/L {1:+0.0000}, PF {2}, W/L {3}/{4}. Loop stopped." -f `
            $state.live_trades, $state.live_pl, $state.profit_factor, $state.live_wins, $state.live_losses)
        break
    }

    # ---- STOP LOSING. Nothing used to halt a losing run. ----
    if ($state -and $null -ne $state.live_pl -and $state.live_pl -le (-1 * $MaxLossUsd)) {
        Write-Banner "HALTED: LOSS LIMIT REACHED" "Red"
        Write-Line ("live P/L {0:+0.0000;-0.0000;0.0000} is at or past the -{1} limit" -f $state.live_pl, $MaxLossUsd) "Red"
        Write-Line ("wins/losses = {0}/{1}   profit factor = {2}" -f $state.live_wins, $state.live_losses, $state.profit_factor) "Red"
        Write-Line "Not continuing to trade a losing book. Investigate before restarting." "Red"
        Send-Milestone -Text (
            "R3V3N!R HALTED: loss limit. P/L {0:+0.0000;-0.0000;0.0000} over {1} trades, W/L {2}/{3}, wallet `${4}. Needs you." -f `
            $state.live_pl, $state.live_trades, $state.live_wins, $state.live_losses, $state.usdc)
        break
    }
    if ($state -and $null -ne $state.usdc -and $state.usdc -lt $MinWalletUsd) {
        Write-Banner "HALTED: WALLET BELOW FLOOR" "Red"
        Write-Line ("deployable stable {0} is under the {1} floor" -f $state.usdc, $MinWalletUsd) "Red"
        Write-Line "Not continuing to spend down the wallet. Investigate before restarting." "Red"
        Send-Milestone -Text (
            "R3V3N!R HALTED: wallet `${0} under `${1} floor. {2} live trades, P/L {3:+0.0000;-0.0000;0.0000}. Needs you." -f `
            $state.usdc, $MinWalletUsd, $state.live_trades, $state.live_pl)
        break
    }

    # ---- progress toward the bar ----
    if ($state -and $state.live_rows -ge 1) {
        Write-Line ("LIVE: {0}/{1} trades  P/L {2:+0.0000;-0.0000;0.0000}  PF {3} (need {4})  W/L {5}/{6} -- continuing" -f `
                    $state.live_trades, $MinLiveTrades, $state.live_pl, `
                    $state.profit_factor, $MinProfitFactor, $state.live_wins, $state.live_losses) "Yellow"
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
    Write-Line "sprint goal: $sprintGoal" "Cyan"
    else          { Write-Line "every link passes; waiting on the executor" "Green" }

    # ---- nudge Claude, and verify it answered ----
    # Choose this pass's sprint objective from the live state.
    #
    # A pass with an open-ended goal ("get to live trading") has no natural
    # stopping point, so it spends the whole pass understanding. A concrete
    # objective with a deadline gives it one, and makes a missed goal
    # visible instead of blurred into "still working on it".
    $sprintGoal = if ($state -and $state.live_trades -ge $MinLiveTrades -and $state.profit_factor -lt $MinProfitFactor) {
        "Raise money_button's live profit factor above $MinProfitFactor within this pass. It has $($state.live_trades) live trades at PF $($state.profit_factor). Find the losing pattern in the closed trades and fix the entry or exit rule that causes it."
    } elseif ($state -and $state.live_rows -ge 1) {
        "Get money_button to $MinLiveTrades profitable live trades. $($state.live_trades) settled so far, P/L $($state.live_pl). Keep them coming and keep them small."
    } elseif ($state -and $state.live_attempts -gt 0) {
        "Turn a blocked live entry into a SETTLED one within this pass. $($state.live_attempts) live attempts exist and NONE settled. Take one blocked attempt, find the exact gate that stopped it, and get a real transaction hash on Base -- dust-sized is fine."
    } else {
        "Get ONE real money_button trade onto the chain within this pass, and paste its transaction hash. Smallest amount that can settle."
    }

    $userNote = Read-Inbox
    $noteBlock = ""
    if ($userNote) {
        $noteBlock = @"

## MESSAGE FROM THE USER -- read this first

The user wrote this while you were working. It takes priority over the
failing-link rule above: address it, or say plainly why you cannot.

$userNote

"@
    }

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
$noteBlock

Work ONLY on that link. Diagnose it with real measurements -- query the DB,
read the code, check on-chain. Never assume; verify. If a number looks too
good, it is probably an artifact: check it against an independent source
before believing it.

Then: fix the root cause, add a test that pins the behaviour, run the
relevant tests, commit, and restart production if needed.

Report concisely: which link failed, the evidence, the fix, and the next link.
If a live trade LOSES, verify the demotion guards fired and report the P/L
honestly -- never hide a loss.

## THIS PASS IS A TIMEBOXED SPRINT

Your objective for this pass, in one line:

    $sprintGoal

Treat that as a hard constraint, not an aspiration. It exists to stop a pass
disappearing into analysis: this loop has run thousands of passes and settled
one on-chain trade, because "understand the problem" has no stopping point
while "make a trade happen" does.

How to run a sprint:

  * Spend the FIRST few minutes deciding the shortest path to the objective,
    then commit to it. If two routes both reach it, take the one you can
    finish, not the one you would prefer to have built.
  * Prefer the change that produces EVIDENCE over the change that produces
    architecture. A dust-sized real trade beats a refactor that would make
    trading nicer later.
  * If you reach the objective early, say so plainly and spend the rest of
    the pass hardening it -- a test that pins it, the callers you did not
    check yet -- rather than starting something new you cannot finish.
  * If you will NOT reach it, say so BEFORE the pass ends. Name the single
    blocker, what you tried, and the smallest next step. A pass that ends
    with "blocked on X, here is the evidence, here is the next move" is
    worth more than one that ends mid-thought.
  * Never fake the objective to close it. Marking a goal met on a dry run, a
    mocked trade, or a check script passing is worse than missing it, and
    this repo has already shipped four strategies whose entire records were
    fabricated. If the honest answer is "not met", that is the answer.

The sprint objective NEVER overrides the correctness rules below. Shipping a
change that breaks another link, or that skips the contract checks, does not
count as reaching it.

## BEFORE YOU CHANGE ANYTHING: what else does this touch?

A fix that repairs one link and breaks another is not progress, it is churn.
This repo's own history is mostly that failure:

  6e44f33  money_button was built on a signal with the WRONG SIGN
  9084f03  an entry price quoted in the WRONG UNITS reached the gate
  ef41710  volatility scored over the wrong WINDOW (six days, not the trade)
  86ae43f  95% of the outcomes that decide graduation were silently LOST
  ab74328  a strategy was convicted on trades it did not make
  aa60c9a  a missing address blocked the swap AND corrupted the observation

Every one passed the check in front of it while being wrong underneath.

So for each change, before you write it:

1. FIND THE CALLERS. grep for every caller of what you are about to change,
   and every reader of any file, table or key you are about to write. Name
   them in your report. If a function is called from five places, your change
   is a five-place change -- make it correctly in all five AT ONCE, across as
   many classes or modules as that takes. A partial change that leaves four
   callers on the old contract is worse than no change, because it looks done.

2. CHECK THE CONTRACT AT EVERY BOUNDARY YOU CROSS. For each input you consume
   and each output you produce, verify by MEASUREMENT, not by reading the code
   and assuming:

     * TYPE   - is it the float/int/str/Decimal/None the other side expects?
                A str "0.05" and a float 0.05 both "work" until they do not.
     * SHAPE  - dict vs list vs scalar; one row vs many; nested vs flat.
                Is an empty result [] or {} or None, and does the caller
                handle the one you actually return?
     * UNITS  - raw base units vs human decimals, fraction vs percent, bps vs
                ratio, seconds vs ms, USD vs token. This repo has already
                shipped a wrong-units price and a wrong-sign signal.
     * RANGE  - can it be negative, zero, NaN, inf, or absent? What does the
                consumer do with each? Say what you checked.
     * TIME   - epoch seconds vs ms, UTC vs local, window start vs end,
                inclusive vs exclusive bounds.

   Print the actual value and its type at the boundary and read it. Do not
   infer it from the signature.

3. PROVE YOU DID NOT BREAK ANYTHING. Run the tests that cover every caller you
   found, not only the one you edited. If no test covers a caller you changed,
   write one. Report what you ran and what passed -- "tests pass" without
   naming them is not evidence.

4. IF IT CANNOT BE DONE WITHOUT BREAKING SOMETHING, SAY SO. Do not ship a
   change that trades one broken link for another and report it as a fix.
   Explain what conflicts, and what the correct cross-cutting change would be.

## PRIORITY: money_button is the most important strategy

Split your effort roughly 50/50 between the failing link above and the
money_button lane. It buys low and sells higher inside 5-30 minutes, which
is the shortest horizon this feed supports and the fastest way to accrue
real evidence. It may become several parallel short-horizon strategies
scheduled through the bus -- that is wanted, not a deviation.

Known money_button problems, measured 2026-09-01 (verify before trusting):

  * It is NOT IN THE LEDGER AT ALL. data/strategy_ledger.json has only
    atf_static, obv_accumulation@1w, rsi_reversal@5h, obv_accumulation@3d.
    The registry claims 76 money_button ghost trades, but the ledger is what
    gates graduation, so money_button can NEVER graduate no matter how well
    it trades. Find why its outcomes do not reach StrategyLedger.record()
    and fix that first -- everything else about this lane is downstream.
  * Its record is 16 wins / 60 losses, -0.3997. Fired often and lost. Read
    the cost gate in trading/strategies/money_button.py before loosening
    anything: firing more is how the previous ledger was destroyed. If the
    edge is not there, say so plainly rather than tuning until it looks good.
  * Its registry entry records no symbols, so per-symbol behaviour cannot be
    analysed. Fix the recording so symbols are captured.

Do not delete or disable money_button to make a gate pass.

## Trust the numbers before you act on them

Four strategies carried fabricated records -- exactly +1.0000 profit per
trade, 100% win rate, no symbols, all written in one six-minute window by a
test. They were purged on 2026-09-01 (scripts/purge_test_artifacts.py).

Before treating ANY strategy record as evidence, check it is a measurement:
real records name symbols, have both wins and losses, and show varied
amounts. If you find another fabricated record, purge it and say so. A
strategy that "passes" on invented numbers is worse than one that fails
honestly, because it spends real money.

atf_static's record was checked and is genuine (244 trades, 120/124 W/L,
real symbol spread) -- do not purge it, but do not treat it as the only
viable lane either.

## Ground rules

Wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad on base.
Keep ENABLE_GAS_REFILL=0 -- the refill drained the wallet three times.

NOTE on live_rows: it counts ATTEMPTS, including live-entry-blocked and
live-dry-run-entry. As of 2026-09-01 all 6 "live" rows are blocked or dry
run -- NO REAL MONEY HAS BEEN SPENT YET. The most recent blocks say
reason=token_unresolved. Do not report live trading as working until a row
exists that actually settled on-chain.

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
    # A real rate_limit_event beats guessing from prose: it carries the exact
    # resetsAt epoch. Treat "blocked" (or exhausted utilization) as a limit
    # even when Claude still produced a reply.
    if ($script:LastRateLimit) {
        $ri = $script:LastRateLimit
        $util = [double]$ri.utilization
        $blocked = ($ri.status -eq "blocked") -or ($util -ge 1.0)
        if ($blocked -and $ri.resetsAt) {
            $resetAt = [DateTimeOffset]::FromUnixTimeSeconds([long]$ri.resetsAt).LocalDateTime
            $wait = [int](($resetAt - (Get-Date)).TotalSeconds) + 60
            if ($wait -gt 0) {
                Write-Banner "OUT OF QUOTA -- $($ri.rateLimitType) EXHAUSTED" "Red"
                Write-Line ("resets {0} ({1:N1} hours from now)" -f `
                            $resetAt.ToString("ddd HH:mm"), ($wait / 3600)) "Yellow"
                Send-MilestoneOnce -Key "quota_blocked" -Text (
                    "R3V3N!R: out of {0} quota, paused until {1}. Loop resumes on its own." -f `
                    $ri.rateLimitType, $resetAt.ToString("ddd HH:mm"))
                Wait-ForQuota -Seconds $wait
                $pass--
                continue
            }
        }
    }

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
