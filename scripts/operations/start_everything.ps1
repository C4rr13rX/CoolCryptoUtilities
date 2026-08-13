$ErrorActionPreference = "Stop"
$CryptoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WizardRoot = "D:\Projects\W1z4rDV1510n"
$ProxyStarter = Join-Path $WizardRoot "scripts\aws\start_programming_brain_proxy.ps1"
$CryptoStarter = Join-Path $CryptoRoot "launch_revenir.ps1"

if (-not (Test-Path -LiteralPath $ProxyStarter)) {
    throw "Senior software brain proxy starter is missing: $ProxyStarter"
}
if (-not (Test-Path -LiteralPath $CryptoStarter)) {
    throw "Crypto stack launcher is missing: $CryptoStarter"
}

Write-Host "[1/2] Ensuring the private senior software brain connection" -ForegroundColor Cyan
& $ProxyStarter
Write-Host "[2/2] Ensuring the crypto brain, Django, pipeline, and production manager" -ForegroundColor Cyan
& $CryptoStarter
