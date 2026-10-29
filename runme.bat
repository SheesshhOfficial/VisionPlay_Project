@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
  echo Cannot open the project folder. Extract it to a local writable folder first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Please extract the entire ZIP, then run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" check_python.py --venv
if errorlevel 1 (
  echo The local environment is invalid. Run setup.bat for repair instructions.
  pause
  exit /b 1
)
set "CAM=0"
if not "%~1"=="" set "CAM=%~1"
".venv\Scripts\python.exe" app.py --camera "%CAM%"
if errorlevel 1 (
  echo.
  echo See the error above and README.md. Try: runme.bat 1
  pause
  exit /b 1
)
