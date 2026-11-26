@echo off
setlocal
set "TRACK2_RUNNER=C:\Users\TAMSystech\AppData\Local\Temp\AirKnifeVendorDemo-i3t6trh5\micromamba.exe"
set "TRACK2_ENV=C:\Users\TAMSystech\AppData\Local\Temp\AirKnifeVendorDemo-i3t6trh5\simenv"
if not exist "%TRACK2_RUNNER%" goto missing
if not exist "%TRACK2_ENV%\python.exe" goto missing
"%TRACK2_RUNNER%" run -p "%TRACK2_ENV%" python -X utf8 "%~dp0source\demo.py" %*
set "TRACK2_EXIT=%ERRORLEVEL%"
if not "%TRACK2_EXIT%"=="0" pause
exit /b %TRACK2_EXIT%
:missing
echo The local demo environment is unavailable. See source\README.md for setup.
pause
exit /b 1
