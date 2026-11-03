@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
title S^&P 500 Balance Sheet Dashboard

rem ---- Python finden (3.10 bis 3.13) --------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo [FEHLER] Python wurde nicht gefunden. Bitte Python 3.11 oder 3.12 von https://www.python.org installieren
  echo          und bei der Installation "Add python.exe to PATH" anhaken.
  pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Erstelle virtuelle Umgebung ...
  %PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" || (
    echo [FEHLER] Python 3.10 oder neuer wird benoetigt. & pause & exit /b 1)
  %PY% -m venv .venv || (echo [FEHLER] Virtuelle Umgebung konnte nicht erstellt werden. & pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"

echo [2/4] Pruefe und installiere Pakete (beim ersten Start einige Minuten) ...
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt || (
  echo [FEHLER] Installation der Pakete fehlgeschlagen - Internetverbindung pruefen. & pause & exit /b 1)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo.
  echo  Bitte in der sich oeffnenden Datei .env bei SEC_USER_AGENT deinen Namen und deine
  echo  E-Mail-Adresse eintragen, speichern und Notepad schliessen. Die SEC verlangt das.
  echo.
  notepad ".env"
)

"%VPY%" -c "from src.config import get_settings; get_settings().require_user_agent()" || (
  echo. & echo [FEHLER] SEC_USER_AGENT in .env ist noch nicht korrekt gesetzt. & notepad ".env" & pause & exit /b 2)

if /i "%~1"=="nodata" goto dashboard
echo [3/4] Starte im Hintergrund: SEC-Datenabruf, danach Kurse + ML-Training ...
start "Datenabruf und ML" /min cmd /c background_update.bat

:dashboard
echo [4/4] Starte Dashboard: http://localhost:8501
echo        Fortschritt des Datenabrufs: Seite "Daten laden". Beenden mit Strg+C.
"%VPY%" -m streamlit run app.py --server.port 8501 --server.headless false
endlocal
