param(
  [string]$PostgresDb = "coolcrypto",
  [string]$PostgresUser = "postgres",
  [string]$PostgresPassword = "postgres",
  [string]$PostgresHost = "127.0.0.1",
  [string]$PostgresPort = "5432"
)

$ErrorActionPreference = 'Stop'

Write-Host "[setup] CoolCryptoUtilities full setup" -ForegroundColor Cyan

# Ensure Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "Python not found. Install Python 3.11+ first." -ForegroundColor Red
  exit 1
}

# Create venv
if (-not (Test-Path .venv)) {
  Write-Host "[setup] Creating venv" -ForegroundColor Yellow
  python -m venv .venv
}

$py = Join-Path (Resolve-Path .venv) "Scripts\python.exe"
& $py -m pip install -U pip

# Install Python deps
Write-Host "[setup] Installing Python deps" -ForegroundColor Yellow
& $py -m pip install -r requirements.txt
& $py -m pip install django psycopg[binary]

# Install Postgres if missing
if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
  Write-Host "[setup] PostgreSQL not found; attempting install" -ForegroundColor Yellow
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id PostgreSQL.PostgreSQL --silent --accept-source-agreements --accept-package-agreements
  } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install postgresql -y
  } else {
    Write-Host "No winget/choco found. Install PostgreSQL manually." -ForegroundColor Red
  }
}

# Init database if possible
if (Get-Command psql -ErrorAction SilentlyContinue) {
  Write-Host "[setup] Initializing database" -ForegroundColor Yellow
  $env:PGHOST = $PostgresHost
  $env:PGPORT = $PostgresPort
  $env:PGUSER = $PostgresUser
  $env:PGPASSWORD = $PostgresPassword
  try {
    & createdb $PostgresDb | Out-Null
  } catch {
    Write-Host "[setup] createdb skipped (already exists or permission)" -ForegroundColor DarkYellow
  }
}

# Frontend deps
if (Test-Path "web\frontend\package.json") {
  if (Get-Command npm -ErrorAction SilentlyContinue) {
    Write-Host "[setup] Installing frontend deps" -ForegroundColor Yellow
    Push-Location web\frontend
    npm install
    Pop-Location
  } else {
    Write-Host "npm not found. Install Node.js to build frontend." -ForegroundColor Red
  }
}

# Migrate Django
Write-Host "[setup] Running migrations" -ForegroundColor Yellow
$env:DJANGO_DB_VENDOR = "postgres"
$env:POSTGRES_DB = $PostgresDb
$env:POSTGRES_USER = $PostgresUser
$env:POSTGRES_PASSWORD = $PostgresPassword
$env:POSTGRES_HOST = $PostgresHost
$env:POSTGRES_PORT = $PostgresPort
& $py web\manage.py migrate

Write-Host "[setup] Done." -ForegroundColor Green
