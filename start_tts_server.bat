@echo off
REM AniVoiceChat - Start GPT-SoVITS TTS Server

set GPT_ROOT=D:\Vscode_Program\GPT-SoVITS
set GPT_PYTHON=%GPT_ROOT%\runtime\python.exe

if not exist "%GPT_ROOT%" (
    echo [ERROR] GPT-SoVITS directory not found: %GPT_ROOT%
    pause
    exit /b 1
)

if not exist "%GPT_PYTHON%" (
    echo [ERROR] GPT-SoVITS Python not found: %GPT_PYTHON%
    pause
    exit /b 1
)

echo [INFO] Changing to GPT-SoVITS directory...
cd /d "%GPT_ROOT%"

echo.
echo [INFO] Starting TTS API server...
echo [INFO] Service URL: http://127.0.0.1:9880
echo [INFO] Python: %GPT_PYTHON%
echo.
echo Press Ctrl+C to stop the server
echo ========================================

"%GPT_PYTHON%" api_v2.py -a 127.0.0.1 -p 9880

pause
