@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0..\..\..\..") do set "DIPLOM_ROOT=%%~fI"
if not exist "%DIPLOM_ROOT%\.venv\Scripts\python.exe" (
    echo .venv not found at "%DIPLOM_ROOT%\.venv"
    echo Create it: py -3.11 -m venv "%DIPLOM_ROOT%\.venv"
    pause
    exit /b 1
)
"%DIPLOM_ROOT%\.venv\Scripts\python.exe" "%~dp0main_demo.py" %*
