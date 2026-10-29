@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
  echo Cannot open the project folder. Extract it to a normal writable folder first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" train_neural.py %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo Neural training did not complete. Review the message above; no accuracy is available unless a real run completes.
pause
exit /b %RESULT%
