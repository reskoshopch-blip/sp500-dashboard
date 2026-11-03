@echo off
rem Wird vom Windows-Zeitplan aufgerufen (ohne Fenster-Pause).
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m src.gex.collect >> "data\gex.log" 2>&1
if exist "cloud.env" ".venv\Scripts\python.exe" -m src.cloud.sync >> "data\cloud.log" 2>&1
