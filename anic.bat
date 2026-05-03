@echo off
REM AniVoiceChat Startup Script
REM Usage: anic.bat command [options]

python __main__.py %*

if %ERRORLEVEL% NEQ 0 (
    exit /b %ERRORLEVEL%
)
