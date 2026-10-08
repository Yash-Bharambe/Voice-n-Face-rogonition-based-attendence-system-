@echo off
title SnapClass - AI Attendance App (Streamlit :8501)
cd /d "%~dp0\app"
echo ========================================================
echo   Starting AI Attendance App on http://localhost:8501
echo ========================================================
if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
) else (
    python -m streamlit run app.py --server.port 8501
)
pause
