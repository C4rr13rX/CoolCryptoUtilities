param(
    [string]$Listen = "127.0.0.1:8765",
    [string]$ServiceName = "C0d3rNativeOsService"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$serviceRoot = Join-Path $root "tools\c0d3rV2\native_os_service"
$runtimeRoot = Join-Path $root "runtime\native_os_service"
$tokenPath = Join-Path $runtimeRoot "token.txt"
$binPath = Join-Path $serviceRoot "target\release\c0d3r-native-os-service.exe"

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $tokenPath)) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    [Convert]::ToHexString($bytes).ToLowerInvariant() | Set-Content -LiteralPath $tokenPath -NoNewline -Encoding ascii
}

Push-Location $serviceRoot
try {
    cargo build --release
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $binPath)) {
    throw "Build did not produce $binPath"
}

[Environment]::SetEnvironmentVariable("C0D3R_NATIVE_OS_URL", "http://$Listen", "User")
[Environment]::SetEnvironmentVariable("C0D3R_NATIVE_OS_TOKEN_FILE", $tokenPath, "User")

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.Status -ne "Stopped") {
        Stop-Service -Name $ServiceName -Force
    }
    sc.exe delete $ServiceName | Out-Null
    Start-Sleep -Seconds 2
}

$quotedBin = '"' + $binPath + '" --service --listen ' + $Listen + ' --token-file "' + $tokenPath + '"'
sc.exe create $ServiceName binPath= $quotedBin start= auto DisplayName= "C0D3R Native OS Service" | Out-Null
sc.exe description $ServiceName "Loopback-only authenticated OS/file command service for C0D3R V2." | Out-Null
Start-Service -Name $ServiceName

Write-Host "Installed and started $ServiceName on http://$Listen"
Write-Host "Token file: $tokenPath"
