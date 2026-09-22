@echo off
title Autonomous Driving Multi-Task Perception Cockpit
echo =======================================================
echo   Launching Autonomous Driving Perception Cockpit...
echo =======================================================
echo.

REM Activate virtual environment if present
if exist "venv311\Scripts\activate.bat" (
    call venv311\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Open default browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

REM Run the FastAPI inference server
python app\server.py

pause
