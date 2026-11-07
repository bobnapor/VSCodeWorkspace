@echo off
setlocal
rem One-time setup of a computer for the daily college football run (README.md, section 4):
rem   1. finds Python 3.8 (install Python 3.8.10 from python.org first if it is missing),
rem   2. installs the packages at the versions the models were built with (requirements.txt),
rem   3. records where this code and that Python are (the CFB_CODE and CFB_PYTHON settings), and
rem   4. puts the CFB folder and "CFB daily run" on the Desktop.
rem Safe to run again (after moving the code, for example).  CollegeFootballData's API key goes in cfbd_key.txt here.
cd /d "%~dp0"
set "PYEXE="
for /f "delims=" %%p in ('py -3.8 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%p"
if defined PYEXE goto found
for /f "delims=" %%p in ('python -c "import sys; print(sys.executable) if sys.version_info[:2] == (3, 8) else None" 2^>nul') do set "PYEXE=%%p"
if defined PYEXE goto found
echo Python 3.8 was not found.  Install Python 3.8.10 (Windows installer) from
echo https://www.python.org/downloads/release/python-3810/ and run setup.bat again.
goto fail

:found
echo Python: %PYEXE%
"%PYEXE%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo The packages could not be installed; see the messages above.
    goto fail
)
set "CODE=%~dp0"
set "CODE=%CODE:~0,-1%"
setx CFB_CODE "%CODE%" >nul
setx CFB_PYTHON "%PYEXE%" >nul
set "CFB_SETUP_CODE=%CODE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d = [Environment]::GetFolderPath('Desktop'); New-Item -ItemType Directory -Force -Path (Join-Path $d 'CFB') | Out-Null; Copy-Item -LiteralPath (Join-Path $env:CFB_SETUP_CODE 'desktop_launcher.bat') -Destination (Join-Path $d 'CFB daily run.bat') -Force"
if errorlevel 1 (
    echo Could not put "CFB daily run" on the Desktop.
    goto fail
)
if not exist cfbd_key.txt echo Put your CollegeFootballData API key in %CODE%\cfbd_key.txt before the first run.
echo.
echo Set up.  Double-click "CFB daily run" on the Desktop once a day, around 10 AM Eastern.
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
