@echo off
title SnapClass - Landing Page (Flask :5002)
cd /d "%~dp0\landing"
echo ========================================================
echo   Starting Landing Page on http://localhost:5002
echo ========================================================
if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe app.py
) else (
    python app.py
)
pause
