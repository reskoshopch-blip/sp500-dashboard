@echo off
rem Nasdaq-IPO-Kalender (3 Jahre Historie) laden, auswerten und in die Cloud laden.
cd /d "%~dp0"
set PYTHONUTF8=1
echo Lade Nasdaq-IPOs ... (erstes Mal einige Minuten)
".venv\Scripts\python.exe" -m src.ipo.nasdaq
if exist "cloud.env" ".venv\Scripts\python.exe" -m src.cloud.sync --only ipo
pause
