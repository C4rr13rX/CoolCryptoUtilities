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

if (-not $Message) {
    Write-Host 'usage: .\Tell.ps1 your message to the agent' -ForegroundColor Yellow
    exit 1
}

$repo  = Split-Path -Parent $MyInvocation.MyCommand.Path
$inbox = Join-Path $repo 'data\agent_inbox.md'
$text  = ($Message -join ' ').Trim()

# Append rather than overwrite: several notes written before the next pass
# should all be delivered, not just the last one.
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $inbox -Encoding UTF8 -Value "[$stamp] $text"

Write-Host 'queued for the next pass:' -ForegroundColor Green
Write-Host "  $text" -ForegroundColor White

$pending = @(Get-Content $inbox -ErrorAction SilentlyContinue).Count
if ($pending -gt 1) {
    Write-Host "($pending messages waiting; all delivered together)" -ForegroundColor DarkGray
}
