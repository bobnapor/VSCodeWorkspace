@echo off
rem The daily college football run: the spread picks, then the day's share card of new picks in CFB\Cards on the
rem Desktop and the card record; see cfb_run_all.py.  Run it once a day, around 10 AM Eastern, with no options
rem ("CFB daily run" on the Desktop calls this).  Uses the Python setup.bat recorded for this computer (CFB_PYTHON),
rem else python, else the py launcher.
cd /d "%~dp0"
if defined CFB_PYTHON if exist "%CFB_PYTHON%" goto recorded
python --version >nul 2>&1
if errorlevel 1 (
    py -3 cfb_run_all.py %*
) else (
    python cfb_run_all.py %*
)
goto done

:recorded
"%CFB_PYTHON%" cfb_run_all.py %*

:done
echo.
pause
