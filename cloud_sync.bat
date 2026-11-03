@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (echo Bitte zuerst start.bat ausfuehren. & pause & exit /b 1)
rem Ergebnisse in die Cloud (Supabase) hochladen. cloud_sync.bat --force laedt alles neu hoch.
".venv\Scripts\python.exe" -m src.cloud.sync %*
pause
