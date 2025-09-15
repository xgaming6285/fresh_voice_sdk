@echo off
REM Simple Windows VoIP Voice Agent Starter
REM This script starts the voice agent without Unicode characters

echo ================================================
echo  Windows VoIP Voice Agent - Starting
echo ================================================
echo  Phone: +359898995151 (Gate VoIP SIM)
echo  Your IP: 192.168.50.158
echo  Gate IP: 192.168.50.127
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found

REM Check if we're in the right directory
if not exist "windows_voice_agent.py" (
    echo [ERROR] windows_voice_agent.py not found
    echo Please run this script from the project directory
    pause
    exit /b 1
)

echo [OK] Project files found

REM Check if .env exists
if not exist ".env" (
    echo [WARNING] .env file not found - running setup first
    python setup_windows_simple.py
    if errorlevel 1 (
        echo [ERROR] Setup failed
        pause
        exit /b 1
    )
)

REM Check if API key is configured
findstr /C:"your_google_ai_studio_api_key_here" .env >nul
if not errorlevel 1 (
    echo.
    echo [WARNING] Google API key not configured!
    echo Please edit .env file and add your API key
    echo Opening .env file now...
    echo.
    notepad .env
    echo.
    echo After adding your API key, run this script again
    pause
    exit /b 1
)

echo [OK] Configuration looks good

REM Start the voice agent
echo.
echo [INFO] Starting Voice Agent Server...
echo.
echo ================================================
echo  Voice Agent API Server
echo ================================================
echo  Health Check: http://localhost:8000/health
echo  API Docs: http://localhost:8000/docs
echo  Configuration: http://localhost:8000/api/config
echo ================================================
echo  Press Ctrl+C to stop the server
echo ================================================
echo.

python windows_voice_agent.py --host 0.0.0.0 --port 8000

echo.
echo [INFO] Voice Agent stopped
pause
