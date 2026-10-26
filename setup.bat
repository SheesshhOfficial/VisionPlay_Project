@echo off
setlocal
cd /d "%~dp0"
echo === VisionPlay setup: local virtual environment only ===
set "PY="
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V check_python.py >nul 2>&1
    if not errorlevel 1 set "PY=py -%%V"
  )
)
if not defined PY (
  python check_python.py >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if not defined PY goto missing
%PY% check_python.py
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" check_python.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" app.py --self-test
if errorlevel 1 goto failed
echo.
echo READY! Double-click runme.bat. No internet is needed to play.
pause
exit /b 0
:missing
echo.
echo No compatible 64-bit Python 3.10, 3.11 or 3.12 was found.
echo Python 3.13 and newer are NOT supported by these pinned dependencies.
echo Your default Python, if available:
py --version 2>nul
python --version 2>nul
echo Install Python 3.11 alongside your existing Python; do not uninstall it.
echo Install 64-bit Python 3.11 from https://www.python.org/downloads/release/python-3119/
echo Enable the Python launcher during installation. Then rerun setup.bat.
pause
exit /b 1
:failed
echo.
echo Setup failed. Read the error above and README.md troubleshooting.
echo Nothing was installed globally. Fix the issue, then rerun setup.bat.
pause
exit /b 1
