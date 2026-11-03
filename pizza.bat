@echo off
rem Pizza-Index abrufen und nur diesen Teil in die Cloud laden.
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m src.geo.pizza >> "data\pizza.log" 2>&1
if exist "cloud.env" ".venv\Scripts\python.exe" -m src.cloud.sync --only geo >> "data\cloud.log" 2>&1
