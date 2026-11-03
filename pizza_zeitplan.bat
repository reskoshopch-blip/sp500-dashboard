@echo off
rem Richtet einen Windows-Zeitplan ein: Pizza-Index alle 30 Minuten (solange der PC laeuft).
cd /d "%~dp0"
schtasks /Create /F /SC MINUTE /MO 30 /TN "Dashboard Pizza-Index" /TR "\"%~dp0pizza.bat\""
if errorlevel 1 (echo Zeitplan konnte nicht erstellt werden.) else (echo Zeitplan erstellt: alle 30 Minuten.)
pause
