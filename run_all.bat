@echo off
title Crypto Bot Launcher
cd /d e:\PythonProjects\VSCodeWorkspace

echo Starting Crypto Signal Bot...
start "Crypto Signal Bot" cmd /k "cd /d e:\PythonProjects\VSCodeWorkspace && C:\Users\Bobby\AppData\Local\Programs\Python\Python38-32\python.exe crypto_main.py"

:: Short pause so the bot initialises before the dashboard tries to read files
timeout /t 3 /nobreak > nul

echo Starting Crypto Dashboard...
start "Crypto Dashboard" cmd /k "cd /d e:\PythonProjects\VSCodeWorkspace && C:\Users\Bobby\AppData\Local\Programs\Python\Python38-32\python.exe crypto_dashboard.py"

echo.
echo Both windows are now running.
echo Close each window individually to stop them.
timeout /t 4 /nobreak > nul
