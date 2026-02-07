@echo off
setlocal enabledelayedexpansion

set REPO_ROOT=%~dp0\..
cd /d "%REPO_ROOT%"

set PY_EXE=
set PY_ARGS=
where py >nul 2>nul && set PY_EXE=py && set PY_ARGS=-3
if not defined PY_EXE (
  where python >nul 2>nul && set PY_EXE=python && set PY_ARGS=
)
if not defined PY_EXE (
  echo Python not found. Install Python 3.11+ first.
  exit /b 1
)

%PY_EXE% %PY_ARGS% - <<PY
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 8):
    print("Python 3.8+ is required. Install a newer Python.", file=sys.stderr)
    raise SystemExit(1)
PY
if errorlevel 1 exit /b 1

if not exist ".venv" (
  echo [quickstart] Creating venv
  %PY_EXE% %PY_ARGS% -m venv .venv
)

set VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe
"%VENV_PY%" -m pip install -U pip
"%VENV_PY%" -m pip install -r requirements.txt
if exist requirements_textbooks.txt (
  "%VENV_PY%" -m pip install -r requirements_textbooks.txt
)

set DJANGO_DB_VENDOR=sqlite
set DJANGO_PREFER_SQLITE_FALLBACK=1

echo [quickstart] Running migrations
"%VENV_PY%" web\manage.py migrate

set BIN_PATH=%REPO_ROOT%\bin
echo %PATH% | find /I "%BIN_PATH%" >nul || set PATH=%BIN_PATH%;%PATH%
powershell -NoProfile -Command "$p=[Environment]::GetEnvironmentVariable('PATH','User'); if ($p -notlike '*%BIN_PATH%*') { [Environment]::SetEnvironmentVariable('PATH', $p+';%BIN_PATH%','User') }" >nul 2>nul

set HOST=127.0.0.1
set PORT=8000
if not "%1"=="" set HOST=%1
if not "%2"=="" set PORT=%2

echo [quickstart] Starting Django at http://%HOST%:%PORT%
"%VENV_PY%" web\manage.py runserver %HOST%:%PORT%
