@echo off
rem The daily college football run, for the Desktop (setup.bat puts it there as "CFB daily run.bat").  It
rem has no computer-specific paths, so it works on every computer that shares this Desktop through OneDrive: setup.bat
rem records where each computer keeps the code (CFB_CODE), and this runs run_all.bat from there.
if not defined CFB_CODE goto notset
if not exist "%CFB_CODE%\run_all.bat" goto notset
call "%CFB_CODE%\run_all.bat" %*
exit /b

:notset
echo This computer is not set up for the college football run yet: run setup.bat in the cfb code folder once.
echo (If you just ran it, sign out of Windows and back in, or restart, and try again.)
echo.
pause
exit /b 1
