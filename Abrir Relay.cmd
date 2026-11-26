@echo off
setlocal
pushd "%~dp0" || exit /b 1
title Relay
set "PYTHONPATH=%~dp0src"
if exist ".venv-relay\Scripts\python.exe" (
    ".venv-relay\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] < (3,12))" >nul 2>nul
    if not errorlevel 1 goto :venv
)
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 goto :launcher
python -c "import sys; sys.exit(sys.version_info[:2] < (3,12))" >nul 2>nul
if not errorlevel 1 goto :python
echo Para abrir Relay, instala Python 3.12 o posterior para Windows.
set "relayExit=1"
goto :done

:venv
".venv-relay\Scripts\python.exe" -B -m relay_router.web_server --open
goto :status
:launcher
py -3.12 -B -m relay_router.web_server --open
goto :status
:python
python -B -m relay_router.web_server --open
:status
set "relayExit=%errorlevel%"
:done
popd
if not "%relayExit%"=="0" pause
exit /b %relayExit%
