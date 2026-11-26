$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv-groq\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Falta el entorno .venv-groq. Ejecuta python -m venv .venv-groq e instala .[api] con ese Python.'
}
Push-Location -LiteralPath $projectDirectory
try {
    & $pythonExecutable (Join-Path $PSScriptRoot 'groq_local.py') @args
    if ($LASTEXITCODE -ne 0) { throw 'La prueba se detuvo. Conserva la salida para revisarla.' }
} finally {
    Pop-Location
}
