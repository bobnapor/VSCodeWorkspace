@echo off
rem The daily NFL run: spread projections, player props, wind unders, then the day's share card of new picks in the
rem 'NFL picks' folder on the Desktop; see nfl_run_all.py.  Run it once a day, around 10 AM Eastern, with no options.
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    py -3 nfl_run_all.py %*
) else (
    python nfl_run_all.py %*
)
echo.
pause
