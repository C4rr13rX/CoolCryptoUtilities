param(
  [string]$OutDir = "dist",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Building BrandDozer CLI (Windows)..."

if (!(Test-Path ".venv")) {
  & $Python -m venv .venv
}

$VenvPython = Join-Path ".venv" "Scripts\\python.exe"
if (!(Test-Path $VenvPython)) {
  throw "Virtualenv python not found at $VenvPython"
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m pip install pyinstaller

& $VenvPython -m PyInstaller --clean --noconfirm scripts\\brandozer.spec

if ($OutDir -ne "dist") {
  if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
  Copy-Item "dist\\brandozer\\brandozer.exe" (Join-Path $OutDir "brandozer.exe") -Force
}

Write-Host "Build complete. Output: dist\\brandozer\\brandozer.exe"
Write-Host "Add to PATH (PowerShell):"
Write-Host "  setx PATH `\"$($PWD)\\dist;`$env:PATH`\""
