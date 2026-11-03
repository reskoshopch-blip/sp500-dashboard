@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (echo Bitte zuerst start.bat ausfuehren. & pause & exit /b 1)
rem Beispiele:  update_data.bat            -> alle Unternehmen, inkrementell
rem             update_data.bat --force    -> alles neu laden
rem             update_data.bat --tickers AAPL,MSFT
".venv\Scripts\python.exe" -m src.pipeline %*
pause
