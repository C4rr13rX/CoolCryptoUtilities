param(
  [string]$Host = "127.0.0.1",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$pyExe = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
  $pyExe = "py"
  $pyArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  $pyExe = "python"
} else {
  Write-Host "Python not found. Install Python 3.11+ first." -ForegroundColor Red
  exit 1
}

& $pyExe @pyArgs - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 8):
    print("Python 3.8+ is required. Install a newer Python.", file=sys.stderr)
    raise SystemExit(1)
PY

if (-not (Test-Path ".venv")) {
  Write-Host "[quickstart] Creating venv" -ForegroundColor Yellow
  & $pyExe @pyArgs -m venv .venv
}

$venvPy = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"
& $venvPy -m pip install -U pip
& $venvPy -m pip install -r requirements.txt
if (Test-Path "requirements_textbooks.txt") {
  & $venvPy -m pip install -r requirements_textbooks.txt
}

$env:DJANGO_DB_VENDOR = "sqlite"
$env:DJANGO_PREFER_SQLITE_FALLBACK = "1"

Write-Host "[quickstart] Running migrations" -ForegroundColor Yellow
& $venvPy web\manage.py migrate

$binPath = (Resolve-Path ".\bin").Path
if ($env:PATH -notlike "*$binPath*") {
  $env:PATH = "$binPath;$env:PATH"
}
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -and $userPath -notlike "*$binPath*") {
  [Environment]::SetEnvironmentVariable("PATH", "$userPath;$binPath", "User")
}

Write-Host "[quickstart] Starting Django at http://$Host`:$Port" -ForegroundColor Green
& $venvPy web\manage.py runserver "$Host`:$Port"
