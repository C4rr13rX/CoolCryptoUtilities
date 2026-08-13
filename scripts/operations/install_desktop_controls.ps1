$ErrorActionPreference = "Stop"
$CryptoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WizardRoot = "D:\Projects\W1z4rDV1510n"
$Desktop = [Environment]::GetFolderPath("Desktop")
$Shell = New-Object -ComObject WScript.Shell

function Install-Shortcut(
    [string]$Name,
    [string]$Script,
    [string]$WorkingDirectory,
    [string]$Description
) {
    if (-not (Test-Path -LiteralPath $Script)) { throw "Shortcut target is missing: $Script" }
    $shortcut = $Shell.CreateShortcut((Join-Path $Desktop "$Name.lnk"))
    $shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $shortcut.Arguments = "-NoLogo -NoExit -ExecutionPolicy Bypass -File `"$Script`""
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,14"
    $shortcut.Save()
}

Install-Shortcut "Start Everything - Wizard Vision" `
    (Join-Path $PSScriptRoot "start_everything.ps1") $CryptoRoot `
    "Start or recover Django, the production manager, both brains, and market evolution."
Install-Shortcut "Monitor Everything - Wizard Vision" `
    (Join-Path $PSScriptRoot "monitor_everything.ps1") $CryptoRoot `
    "Monitor the entire Wizard Vision crypto and programming stack."
Install-Shortcut "Start Crypto Stack" `
    (Join-Path $CryptoRoot "launch_revenir.ps1") $CryptoRoot `
    "Start or recover the crypto brain, pipeline, Django, production manager, and evolution."
Install-Shortcut "Tail Crypto Brain" `
    (Join-Path $PSScriptRoot "monitor_everything.ps1") $CryptoRoot `
    "Tail crypto-brain and production health without stopping services when closed."
Install-Shortcut "Start Senior Software Brain" `
    (Join-Path $WizardRoot "scripts\aws\start_programming_brain_proxy.ps1") $WizardRoot `
    "Start the private local connection to the AWS senior software engineer brain."
Install-Shortcut "Tail Senior Software Brain" `
    (Join-Path $WizardRoot "scripts\aws\show_programming_brain_watch.ps1") $WizardRoot `
    "Start supervision if needed and tail AWS senior software engineer brain progress."

Write-Host "Installed six Wizard Vision start/monitor controls on $Desktop." -ForegroundColor Green
