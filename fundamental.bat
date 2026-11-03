@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (echo Bitte zuerst start.bat ausfuehren. & pause & exit /b 1)
rem Beispiele:  fundamental.bat            -> Makro/COT/Gewinne laden und Fundamental-Terminal berechnen
rem             fundamental.bat --offline  -> nur mit vorhandenen Daten neu rechnen
".venv\Scripts\python.exe" -m src.fundamental.run %*
pause
