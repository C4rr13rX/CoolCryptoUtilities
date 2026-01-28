param(
  [string]$InstallDir = "$env:LOCALAPPDATA\\Brandozer",
  [string]$ExePath = "$PSScriptRoot\\..\\dist\\brandozer\\brandozer.exe",
  [switch]$ForceInstallCodex
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ExePath)) {
  throw "Executable not found at $ExePath. Run build_brandozer_windows.ps1 first."
}

if (!(Test-Path $InstallDir)) {
  New-Item -ItemType Directory -Path $InstallDir | Out-Null
}

$TargetExe = Join-Path $InstallDir "brandozer.exe"
Copy-Item $ExePath $TargetExe -Force

$CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($CurrentPath -notlike "*$InstallDir*") {
  [Environment]::SetEnvironmentVariable("PATH", "$InstallDir;$CurrentPath", "User")
  Write-Host "Added $InstallDir to User PATH. Open a new PowerShell window to use brandozer."
} else {
  Write-Host "Install directory already on PATH."
}

Write-Host "Installed: $TargetExe"

function Test-Codex {
  $cmd = Get-Command codex -ErrorAction SilentlyContinue
  if ($cmd) { return $true }
  return $false
}

function Try-AddCodexToPath {
  $candidates = @(
    "$env:LOCALAPPDATA\\Programs\\codex\\codex.exe",
    "$env:LOCALAPPDATA\\Programs\\OpenAI\\codex\\codex.exe",
    "$env:USERPROFILE\\AppData\\Roaming\\Python\\Python*\\Scripts\\codex.exe",
    "$env:LOCALAPPDATA\\Programs\\Python\\Python*\\Scripts\\codex.exe"
  )
  foreach ($pattern in $candidates) {
    $matches = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($exe in $matches) {
      $dir = Split-Path $exe.FullName -Parent
      $current = [Environment]::GetEnvironmentVariable("PATH", "User")
      if ($current -notlike "*$dir*") {
        [Environment]::SetEnvironmentVariable("PATH", "$dir;$current", "User")
        Write-Host "Added Codex CLI directory to PATH: $dir"
      }
      return $true
    }
  }
  return $false
}

function Try-InstallCodex {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host "Attempting Codex install via winget..."
    winget install --silent --accept-package-agreements --accept-source-agreements OpenAI.Codex | Out-Null
    return $true
  }
  if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host "Attempting Codex install via choco..."
    choco install codex -y | Out-Null
    return $true
  }
  return $false
}

if (-not (Test-Codex)) {
  Write-Host "Codex CLI not found on PATH. Attempting to locate it..."
  $fixed = Try-AddCodexToPath
  if (-not $fixed -and $ForceInstallCodex) {
    $installed = Try-InstallCodex
    if ($installed) {
      $fixed = Try-AddCodexToPath
    }
  }
  if ($fixed) {
    Write-Host "Codex CLI path added. Open a new PowerShell window and run: codex --version"
  } else {
    Write-Host "Codex CLI still not found."
    Write-Host "Resolution: install the OpenAI Codex CLI and ensure `codex` is on PATH."
  }
} else {
  Write-Host "Codex CLI detected: $(Get-Command codex | Select-Object -ExpandProperty Source)"
}

Write-Host "Running post-install self-check..."
try {
  & $TargetExe --help | Out-Null
  Write-Host "Self-check OK: brandozer --help"
} catch {
  Write-Host "Self-check failed: brandozer --help"
  Write-Host "Reason: $($_.Exception.Message)"
  Write-Host "Resolution: ensure Python deps were packaged and that runtime dirs are writable."
}
