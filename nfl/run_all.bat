@echo off
rem The daily NFL run: spread projections, player props, wind unders, then the day's share card of new picks in
rem NFL\Cards on the Desktop; see nfl_run_all.py.  Run it once a day, around 10 AM Eastern, with no options
rem ("NFL daily run" on the Desktop calls this).  Uses the Python setup.bat recorded for this computer (NFL_PYTHON),
rem else python, else the py launcher.
cd /d "%~dp0"
if defined NFL_PYTHON if exist "%NFL_PYTHON%" goto recorded
python --version >nul 2>&1
if errorlevel 1 (
    py -3 nfl_run_all.py %*
) else (
    python nfl_run_all.py %*
)
goto done

:recorded
"%NFL_PYTHON%" nfl_run_all.py %*

:done
echo.
pause
