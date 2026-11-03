@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (echo Bitte zuerst start.bat ausfuehren. & pause & exit /b 1)
rem GEX abrufen und KI aktualisieren. gex.bat --force ignoriert die 6-Stunden-Sperre.
".venv\Scripts\python.exe" -m src.gex.collect %*
pause
