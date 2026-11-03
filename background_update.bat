@echo off
rem Wird von start.bat im Hintergrund gestartet: Bilanzdaten, Kurse + ML, Fundamental, COT.
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m src.pipeline >> "data\pipeline.log" 2>&1
".venv\Scripts\python.exe" -m src.ml.run >> "data\ml.log" 2>&1
".venv\Scripts\python.exe" -m src.fundamental.run >> "data\fundamental.log" 2>&1
".venv\Scripts\python.exe" -m src.cot.run >> "data\cot.log" 2>&1
".venv\Scripts\python.exe" -m src.gex.collect >> "data\gex.log" 2>&1
".venv\Scripts\python.exe" -m src.geo.pizza >> "data\pizza.log" 2>&1
if exist "cloud.env" ".venv\Scripts\python.exe" -m src.cloud.sync >> "data\cloud.log" 2>&1
