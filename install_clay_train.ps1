[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $env:USERPROFILE ".local\bin"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$entry = Join-Path $repo "clay_train_cli.py"
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "Clay training entry point is missing: $entry"
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$launcher = Join-Path $Destination "clay.train.cmd"
$content = @"
@echo off
python "$entry" %*
"@

if ((Test-Path -LiteralPath $launcher) -and -not $Force) {
    $current = Get-Content -LiteralPath $launcher -Raw
    if ($current -ne $content) {
        throw "Refusing to replace a different launcher at $launcher. Re-run with -Force."
    }
}

Set-Content -LiteralPath $launcher -Value $content -Encoding ascii -NoNewline
Write-Output "installed $launcher"
Write-Output "open a new terminal or run: clay.train"
