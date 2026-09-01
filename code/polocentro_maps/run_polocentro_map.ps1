param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PlotArguments
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'plot_polocentro_amc.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Workspace Python was not found at $python"
}

Push-Location $projectRoot
try {
    & $python $script @PlotArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
