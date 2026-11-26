param(
    [string]$Task = 'Corrige normalize_labels: elimina espacios exteriores y entradas vacías, normaliza con casefold y elimina duplicados conservando el orden de primera aparición. No modifiques la lista recibida.',
    [string]$Config = '',
    [string]$Apply = '',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$relayPython = Join-Path $PSScriptRoot '.venv-groq\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $relayPython)) {
    throw 'No se encontró .venv-groq. Consulta docs/relay_mantenimiento.md para usar otro entorno.'
}
$relayTool = Join-Path $PSScriptRoot 'tools\relay_maintain.py'
if ($Apply) {
    & $relayPython $relayTool apply $Apply
} else {
    if (-not $Config) { $Config = Join-Path $PSScriptRoot 'configs\maintenance_demo.json' }
    if ($Check) {
        & $relayPython $relayTool check --config $Config
    } else {
        $relayOutput = Join-Path $PSScriptRoot ('outputs\maintenance\' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
        & $relayPython $relayTool run --config $Config --task $Task --output $relayOutput
    }
}
exit $LASTEXITCODE
