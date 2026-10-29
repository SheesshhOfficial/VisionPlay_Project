@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
  echo Cannot open the project folder. Extract it to a local writable folder first.
  pause
  exit /b 1
)
echo === VisionPlay setup: local virtual environment only ===
rem Reuse only a working, compatible virtual environment. Never delete user files.
if exist ".venv" goto check_existing
set "PY_EXE="
set "PY_ARG="
for %%V in (3.11 3.12 3.10) do call :try_python "py" "-%%V"
rem Official per-user installs can exist without the launcher or a PATH entry.
for %%V in (311 312 310) do call :try_python "%LocalAppData%\Programs\Python\Python%%V\python.exe" ""
rem Inspect every Python executable on PATH, not only a possible Store alias.
for /f "delims=" %%P in ('where python.exe 2^>nul') do call :try_python "%%P" ""
if not defined PY_EXE goto missing
"%PY_EXE%" %PY_ARG% check_python.py
if errorlevel 1 goto failed
"%PY_EXE%" %PY_ARG% -m venv .venv
if errorlevel 1 goto failed
:check_existing
if not exist ".venv\Scripts\python.exe" goto broken_venv
".venv\Scripts\python.exe" check_python.py --venv
if errorlevel 1 goto broken_venv
".venv\Scripts\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 goto broken_venv
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
:try_python
if defined PY_EXE exit /b 0
"%~1" %~2 check_python.py >nul 2>&1
if errorlevel 1 exit /b 0
set "PY_EXE=%~1"
set "PY_ARG=%~2"
exit /b 0
:missing
echo.
echo No compatible 64-bit Python 3.10, 3.11 or 3.12 was found.
echo Python 3.13 and newer are NOT supported by these pinned dependencies.
echo Your default Python, if available:
py --version 2>nul
python --version 2>nul
echo A venv cannot install a missing Python version.
echo Install Python 3.11 alongside your existing Python; do not uninstall it.
echo Install 64-bit Python 3.11 from https://www.python.org/downloads/release/python-3119/
echo Enable the Python launcher during installation. Then rerun setup.bat.
pause
exit /b 1
:broken_venv
echo.
echo The existing .venv is incomplete, incompatible, or no longer works.
echo Nothing has been deleted or replaced.
echo Close VisionPlay. Rename this folder's .venv to .venv-backup
echo using File Explorer, then rerun setup.bat to create a fresh environment.
echo Your captured samples in data and downloaded model in models are kept.
pause
exit /b 1
:failed
echo.
echo Setup failed. Read the error above and README.md troubleshooting.
echo Nothing was installed globally. Fix the issue, then rerun setup.bat.
pause
exit /b 1
