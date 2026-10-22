@echo off
setlocal
rem One-time setup of a computer for the daily NFL run (README.md: "Running on another computer"):
rem   1. finds Python 3.8 (install Python 3.8.10 from python.org first if it is missing),
rem   2. installs the packages at the versions the models were built with (requirements.txt),
rem   3. records where this code and that Python are (the NFL_CODE and NFL_PYTHON settings), and
rem   4. puts the NFL folder and "NFL daily run" on the Desktop.
rem Safe to run again (after moving the code, for example).
cd /d "%~dp0"
set "PYEXE="
for /f "delims=" %%p in ('py -3.8 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%p"
if defined PYEXE goto found
for /f "delims=" %%p in ('python -c "import sys; print(sys.executable) if sys.version_info[:2] == (3, 8) else None" 2^>nul') do set "PYEXE=%%p"
if defined PYEXE goto found
echo Python 3.8 was not found.  Install Python 3.8.10 (Windows installer, 64-bit) from
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
setx NFL_CODE "%CODE%" >nul
setx NFL_PYTHON "%PYEXE%" >nul
set "NFL_SETUP_CODE=%CODE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d = [Environment]::GetFolderPath('Desktop'); New-Item -ItemType Directory -Force -Path (Join-Path $d 'NFL') | Out-Null; Copy-Item -LiteralPath (Join-Path $env:NFL_SETUP_CODE 'desktop_launcher.bat') -Destination (Join-Path $d 'NFL daily run.bat') -Force"
if errorlevel 1 (
    echo Could not put "NFL daily run" on the Desktop.
    goto fail
)
echo.
echo Set up.  Double-click "NFL daily run" on the Desktop once a day, around 10 AM Eastern
echo (the schedule is at the top of README.md in this folder).
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
