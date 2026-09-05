# Restore D:\Projects\W1z4rDV1510n market evolution exactly as it was found
# 2026-09-05, paused to free RAM for a wizard-brain training run.
#
# The watchdog respawns the service on its own (30s restart delay), so this
# only needs to relaunch the WATCHDOG with the arguments it had.
$ErrorActionPreference = "Stop"

$already = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*market_evolution_watchdog*" }
if ($already) {
    Write-Output "watchdog already running (pid $($already.ProcessId)) - nothing to do"
    exit 0
}

$py   = "C:\Python313\python.exe"
$args = @(
    "-u", "D:\Projects\W1z4rDV1510n\scripts\market_evolution_watchdog.py",
    "--python", "C:\Python313\python.exe",
    "--service", "D:\Projects\W1z4rDV1510n\scripts\market_evolution_service.py",
    "--state-dir", "D:\Projects\W1z4rDV1510n\runtime\market-evolution",
    "--min-free-memory-gb", "3.5",
    "--memory-poll-seconds", "15",
    "--restart-delay-seconds", "30",
    "--",
    "--population", "8",
    "--workers", "1",
    "--brain-gate-every", "1",
    "--test-days", "28"
)

Start-Process -FilePath $py -ArgumentList $args `
    -WorkingDirectory "D:\Projects\W1z4rDV1510n" -WindowStyle Hidden
Start-Sleep -Seconds 3
$now = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*market_evolution_watchdog*" }
if ($now) { Write-Output "RESTORED: watchdog pid $($now.ProcessId)" }
else       { Write-Output "FAILED to restore - start it by hand" }
