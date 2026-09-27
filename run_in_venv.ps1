# PowerShell launcher script for executing Python scripts within .venv
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment Python executable not found at: $VenvPython"
    exit 1
}

& $VenvPython @Args
