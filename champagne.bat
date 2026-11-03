@echo off
rem Champagner-Index berechnen und in die Cloud laden.
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m src.geo.champagne >> "data\champagne.log" 2>&1
if exist "cloud.env" ".venv\Scripts\python.exe" -m src.cloud.sync --only geo >> "data\cloud.log" 2>&1
