$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv-groq\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Falta el entorno .venv-groq con las dependencias del proyecto.'
}
Push-Location -LiteralPath $projectDirectory
try {
    & $pythonExecutable (Join-Path $PSScriptRoot 'relay_trajectory_local.py') @args
    if ($LASTEXITCODE -ne 0) { throw 'La recolección se detuvo; conserva los archivos para auditarla.' }
} finally {
    Pop-Location
}
