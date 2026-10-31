$ErrorActionPreference = "Stop"
$CryptoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WizardRoot = "D:\Projects\W1z4rDV1510n"
$Desktop = [Environment]::GetFolderPath("Desktop")
$LocalDesktop = Join-Path $env:USERPROFILE "Desktop"
$RevenirIcon = Join-Path $CryptoRoot "web\static\assets\favicon.ico"
$Shell = New-Object -ComObject WScript.Shell

function Install-Shortcut(
    [string]$Name,
    [string]$Script,
    [string]$WorkingDirectory,
    [string]$Description,
    [string]$IconLocation = "$env:SystemRoot\System32\shell32.dll,14",
    [string]$Arguments = "",
    [string]$TargetFolder = $Desktop
) {
    if (-not (Test-Path -LiteralPath $Script)) { throw "Shortcut target is missing: $Script" }
    if (-not (Test-Path -LiteralPath $TargetFolder)) { return }
    $shortcut = $Shell.CreateShortcut((Join-Path $TargetFolder "$Name.lnk"))
    $shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    if ($Arguments) {
        $shortcut.Arguments = $Arguments
    } else {
        $shortcut.Arguments = "-NoLogo -NoExit -ExecutionPolicy Bypass -File `"$Script`""
    }
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    $shortcut.IconLocation = $IconLocation
    $shortcut.Save()
}

$LauncherScript = Join-Path $CryptoRoot "launch_revenir.ps1"
$RevenirIconSpec = if (Test-Path -LiteralPath $RevenirIcon) { "$RevenirIcon,0" } else { "$env:SystemRoot\System32\shell32.dll,14" }
$RevenirDesc = "R3V3N!R Control Tower -- starts W1z4rDV1510n brain (:8090), Django Waitress (:8001), EdgeForge 3-worker GA, Production Manager, Brain Feeder, Market Evolution, TradingAgent (AGY), and Strategy Evolution."

foreach ($dir in @($Desktop, $LocalDesktop) | Select-Object -Unique) {
    Install-Shortcut "R3V3N!R Control Tower" `
        $LauncherScript $CryptoRoot `
        $RevenirDesc `
        $RevenirIconSpec `
        "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$LauncherScript`"" `
        $dir
}

Install-Shortcut "Start Everything - Wizard Vision" `
    (Join-Path $PSScriptRoot "start_everything.ps1") $CryptoRoot `
    "Start or recover Django, EdgeForge GA, the production manager, both brains, AGY agents, and market evolution." `
    $RevenirIconSpec
Install-Shortcut "Monitor Everything - Wizard Vision" `
    (Join-Path $PSScriptRoot "monitor_everything.ps1") $CryptoRoot `
    "Monitor the entire Wizard Vision crypto, EdgeForge GA, AGY agent, and programming stack."
Install-Shortcut "Start Crypto Stack" `
    $LauncherScript $CryptoRoot `
    $RevenirDesc `
    $RevenirIconSpec
Install-Shortcut "Tail Crypto Brain" `
    (Join-Path $PSScriptRoot "monitor_everything.ps1") $CryptoRoot `
    "Tail crypto-brain and production health without stopping services when closed."
Install-Shortcut "Start Senior Software Brain" `
    (Join-Path $WizardRoot "scripts\aws\start_programming_brain_proxy.ps1") $WizardRoot `
    "Start the private local connection to the AWS senior software engineer brain."
Install-Shortcut "Tail Senior Software Brain" `
    (Join-Path $WizardRoot "scripts\aws\show_programming_brain_watch.ps1") $WizardRoot `
    "Start supervision if needed and tail AWS senior software engineer brain progress."

Write-Host "Installed R3V3N!R Control Tower and Wizard Vision start/monitor controls on $Desktop." -ForegroundColor Green
