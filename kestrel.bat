@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0kestrel.ps1" %*
exit /b %ERRORLEVEL%
