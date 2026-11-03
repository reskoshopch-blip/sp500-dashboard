@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Richtet einen Windows-Zeitplan ein: Montag bis Freitag um 22:15 GEX abrufen (nach US-Boersenschluss).
echo Der PC muss zu dieser Zeit eingeschaltet sein.
schtasks /Create /F /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 22:15 /TN "GEX sammeln Dashboard" /TR "\"%~dp0gex_auto.bat\""
if errorlevel 1 (echo Einrichten fehlgeschlagen.) else (echo Fertig. Entfernen mit: schtasks /Delete /TN "GEX sammeln Dashboard" /F)
pause
