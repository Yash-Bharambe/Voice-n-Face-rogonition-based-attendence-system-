@echo off
title SnapClass AI - Full Stack Launcher
cd /d "%~dp0"
echo ========================================================
echo   Starting SnapClass AI Attendance System (All Services)
echo ========================================================
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe run.py
) else (
    python run.py
)
pause
