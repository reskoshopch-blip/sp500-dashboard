@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m src.verify --sample 15
pause
