<#
.SYNOPSIS
    Send a message to the running trading loop.

.DESCRIPTION
    Appends to data\agent_inbox.md, which the loop reads at the top of its
    next pass, injects into that pass's prompt, and archives. Write any time
    -- including mid-pass. The note lands on the next pass, so there is no
    need to stop the loop to steer it.

.EXAMPLE
    .\Tell.ps1 0x is dead, use the on-chain routes and do not pay for 0x

.EXAMPLE
    .\Tell.ps1 "quoted works too"
#>

# Remaining-arguments only. Adding ValueFromPipeline to this same parameter
# broke positional binding (the message bound to nothing), and reading
# $input blocks when stdin is not redirected -- so keep it simple.
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Message
)

# ValueFromRemainingArguments swallows switches too, so "-Standing" arrives as
# the first word of the message instead of binding to the parameter. Detect it
# here and strip it, so `.\Tell.ps1 -Standing text` behaves the way it reads.
if ($Message -and $Message.Count -gt 0 -and $Message[0] -match '^-+[Ss]tanding$') {
    $Standing = $true
    if ($Message.Count -gt 1) { $Message = $Message[1..($Message.Count - 1)] }
    else { $Message = @() }
}

if (-not $Message) {
    Write-Host 'usage: .\Tell.ps1 [-Standing] your message to the agent' -ForegroundColor Yellow
    Write-Host '  -Standing makes it apply to EVERY pass instead of just the next one.' -ForegroundColor DarkGray
    exit 1
}

$repo  = Split-Path -Parent $MyInvocation.MyCommand.Path
$inbox = if ($Standing) { Join-Path $repo 'data\agent_standing_orders.md' }
         else           { Join-Path $repo 'data\agent_inbox.md' }
$text  = ($Message -join ' ').Trim()

# Append rather than overwrite: several notes written before the next pass
# should all be delivered, not just the last one.
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $inbox -Encoding UTF8 -Value "[$stamp] $text"

Write-Host $(if ($Standing) { 'added to STANDING ORDERS (applies to every pass):' }
             else { 'queued for the next pass:' }) -ForegroundColor Green
Write-Host "  $text" -ForegroundColor White

$pending = @(Get-Content $inbox -ErrorAction SilentlyContinue).Count
if ($pending -gt 1) {
    Write-Host "($pending messages waiting; all delivered together)" -ForegroundColor DarkGray
}
