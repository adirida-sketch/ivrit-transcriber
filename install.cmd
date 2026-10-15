@echo off
rem Double-click to install. Pass a folder to install somewhere else:  install.cmd D:\Transcriber
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -InstallDir "%~1"
)
if errorlevel 1 (
  echo.
  echo The installation did not complete - see the messages above.
  pause
) else (
  echo.
  echo This window will close in 5 seconds...
  "%SystemRoot%\System32\timeout.exe" /t 5 >nul
)
