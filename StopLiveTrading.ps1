<#
.SYNOPSIS
    Stop the GetToLiveTrading loop AND any Claude process it spawned.

.DESCRIPTION
    Closing the loop's window is not enough. Start-Process children outlive
    their parent on Windows, so a force-killed loop leaves a claude process
    running with nothing reading its output -- it keeps working, and keeps
    spending tokens, invisibly.

    Use this instead of closing the window or Stop-Process.
#>

[CmdletBinding()]
param()

$stopped = 0

# 1. The loop itself.
Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*GetToLiveTrading*' } |
    ForEach-Object {
        Write-Host "stopping loop PID $($_.ProcessId)" -ForegroundColor Yellow
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped++
    }

Start-Sleep -Seconds 2

# 2. Any Claude it spawned, now parentless.
#
# Claude runs as claude.exe here, not node.exe -- checking only node.exe left
# 14 claude.exe processes alive after the loop was killed, which is exactly
# the orphan case this script exists to prevent.
#
# The parent check is what keeps this safe: an interactive Claude (yours, or
# a VS Code extension) has a live parent -- Code.exe, a terminal -- so it is
# never touched. Only a Claude whose parent is gone gets reaped.
foreach ($procName in @("claude.exe", "node.exe")) {
    Get-CimInstance Win32_Process -Filter "Name='$procName'" -ErrorAction SilentlyContinue |
        Where-Object { $procName -eq 'claude.exe' -or $_.CommandLine -like '*claude*' } |
        ForEach-Object {
            $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.ParentProcessId)" -ErrorAction SilentlyContinue
            if (-not $owner) {
                Write-Host "stopping orphaned $procName PID $($_.ProcessId)" -ForegroundColor Yellow
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                $stopped++
            }
        }
}

# 3. Console windows left behind by the processes we just killed.
#
# A conhost whose parent is gone keeps painting its last frame, so a dead
# loop still LOOKS like a running one. That is worse than no window: it
# reads as "you left the other one running" when nothing is running at all.
Start-Sleep -Seconds 1
Get-CimInstance Win32_Process -Filter "Name='conhost.exe'" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.ParentProcessId)" -ErrorAction SilentlyContinue
        if (-not $parent) {
            Write-Host "closing orphaned console window PID $($_.ProcessId)" -ForegroundColor Yellow
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }

if ($stopped -eq 0) {
    Write-Host "nothing running." -ForegroundColor Green
} else {
    Write-Host "stopped $stopped process(es)." -ForegroundColor Green
}

$m = New-Object System.Threading.Mutex($false, "Global\R3V3N1R_GetToLiveTrading")
if ($m.WaitOne(0)) { Write-Host "mutex released - clear to restart." -ForegroundColor Green; $m.ReleaseMutex() }
else { Write-Host "WARNING: mutex still held." -ForegroundColor Red }
