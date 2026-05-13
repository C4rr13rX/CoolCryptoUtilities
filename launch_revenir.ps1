# R3V3N!R Control Tower launcher — production WSGI (waitress)
# Replaces `manage.py runserver --noreload`, which is Django's dev server
# and is documented as unsuitable for long-running use (single-threaded,
# silently dies on unhandled exceptions, no graceful restart).  Waitress
# is multi-threaded, kernel-validated for Windows, and keeps the panel
# responsive even when one request stalls.
$projectRoot = "D:\Projects\CoolCryptoUtilities"
$python      = "$projectRoot\.venv\Scripts\python.exe"
$webRoot     = "$projectRoot\web"
$host_       = "127.0.0.1"
$port        = 8000
$threads     = 8   # concurrent worker threads — bumps responsiveness under load

# Health-probe first so we don't double-launch.
$running = $false
try {
    $r = Invoke-WebRequest -Uri "http://${host_}:${port}/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    $running = $true
} catch {}

if (-not $running) {
    Write-Host "Starting R3V3N!R Control Tower via waitress..."
    # run_waitress.py mirrors manage.py's boot sequence (sys.path,
    # EnvLoader, dev defaults) then hands off to waitress.serve.
    $env:WAITRESS_HOST    = $host_
    $env:WAITRESS_PORT    = "$port"
    $env:WAITRESS_THREADS = "$threads"
    $args = @("run_waitress.py")
    Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $webRoot -WindowStyle Minimized

    # Wait up to 30s for the server to accept connections.
    $max = 60
    $i = 0
    while ($i -lt $max) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-WebRequest -Uri "http://${host_}:${port}/" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop | Out-Null
            break
        } catch {}
        $i++
    }
}

Start-Process "http://${host_}:${port}/"
