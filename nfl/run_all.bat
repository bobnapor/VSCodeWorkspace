@echo off
rem One click for the weekly NFL flows (spread projections, player props, wind unders); see nfl_run_all.py.
rem Arguments pass through, e.g.  run_all.bat projections wind
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    py -3 nfl_run_all.py %*
) else (
    python nfl_run_all.py %*
)
echo.
pause
