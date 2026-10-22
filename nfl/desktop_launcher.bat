@echo off
rem The daily NFL run, for the Desktop (setup.bat puts it there as "NFL daily run.bat").  It has no computer-specific
rem paths, so it works on every computer that shares this Desktop through OneDrive: setup.bat records where each
rem computer keeps the code (NFL_CODE), and this runs run_all.bat from there.
if not defined NFL_CODE goto notset
if not exist "%NFL_CODE%\run_all.bat" goto notset
call "%NFL_CODE%\run_all.bat" %*
exit /b

:notset
echo This computer is not set up for the NFL run yet: run setup.bat in the nfl code folder once.
echo (If you just ran it, sign out of Windows and back in, or restart, and try again.)
echo.
pause
exit /b 1
