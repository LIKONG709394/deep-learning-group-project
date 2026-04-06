@echo off
setlocal
cd /d "%~dp0.."
if not exist "%CD%\scripts\run_web.py" (
  echo Could not find scripts\run_web.py from "%CD%"
  exit /b 1
)
if exist "%CD%\.venv\Scripts\python.exe" (
  "%CD%\.venv\Scripts\python.exe" "%CD%\scripts\run_web.py" %*
) else (
  echo No .venv\Scripts\python.exe in "%CD%"
  echo Create venv: python -m venv .venv
  echo Then: .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
