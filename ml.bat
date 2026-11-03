@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (echo Bitte zuerst start.bat ausfuehren. & pause & exit /b 1)
rem Beispiele:  ml.bat                  -> Kurse aktualisieren und Modelle trainieren
rem             ml.bat --skip-prices    -> nur neu trainieren
".venv\Scripts\python.exe" -m src.ml.run %*
pause
