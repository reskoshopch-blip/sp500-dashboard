@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (echo Bitte zuerst start.bat ausfuehren. & pause & exit /b 1)
rem COT-Modell: cot.bat  (oder cot.bat --offline)
".venv\Scripts\python.exe" -m src.cot.run %*
pause
