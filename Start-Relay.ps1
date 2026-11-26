param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501,
    [switch]$OpenCode,
    [switch]$OpenChat,
    [switch]$Investigacion
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$localPython = Join-Path $PSScriptRoot '.venv-relay\Scripts\python.exe'
$pythonArgs = @()
if (Test-Path -LiteralPath $localPython) {
    $pythonCommand = $localPython
} elseif ((Get-Command py -ErrorAction SilentlyContinue) -and (& py -3.12 -c 'print(1)' 2>$null)) {
    $pythonCommand = 'py'
    $pythonArgs = @('-3.12')
} else {
    $pythonCommand = 'python'
}
if ($Investigacion) {
    $scriptPath = Join-Path $PSScriptRoot 'app\relay_app.py'
    & $pythonCommand @pythonArgs -m streamlit run $scriptPath --server.address 127.0.0.1 --server.port $Port
    exit $LASTEXITCODE
}

$serverArgs = @('-B', '-m', 'relay_router.web_server', '--port', $Port)
if ($OpenChat) { $serverArgs += '--open' }
if (-not ($OpenChat -or $OpenCode)) { $serverArgs += '--documents' }
& $pythonCommand @pythonArgs @serverArgs
exit $LASTEXITCODE
