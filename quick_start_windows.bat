@echo off
REM Quick Start Script for Windows VoIP Voice Agent
REM This script will set up and start the Gemini AI voice agent

echo.
echo ================================================
echo  Windows VoIP Voice Agent - Quick Start
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

REM Install dependencies
echo.
echo [INFO] Installing Python dependencies...
pip install -r requirements_windows.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [OK] Dependencies installed

REM Run setup
echo.
echo [INFO] Running Windows setup...
python setup_windows_simple.py
if errorlevel 1 (
    echo [ERROR] Setup failed
    pause
    exit /b 1
)

echo [OK] Setup complete

REM Check if .env exists and has API key
if not exist ".env" (
    echo [ERROR] .env file not found
    echo Please run setup_windows_simple.py first
    pause
    exit /b 1
)

findstr /C:"your_google_ai_studio_api_key_here" .env >nul
if not errorlevel 1 (
    echo.
    echo [WARNING] IMPORTANT: You need to add your Google API key!
    echo.
    echo 1. Get API key from: https://ai.google.dev/
    echo 2. Edit .env file and replace: your_google_ai_studio_api_key_here
    echo 3. Run this script again
    echo.
    notepad .env
    pause
    exit /b 1
)

REM Run tests
echo.
echo [INFO] Running system tests...
python test_windows_voip.py --quick
if errorlevel 1 (
    echo [WARNING] Some tests failed, but continuing...
    timeout /t 3 /nobreak >nul
)

REM Start the voice agent
echo.
echo [INFO] Starting Voice Agent...
echo.
echo ================================================
echo  Voice Agent Starting
echo ================================================
echo  API: http://localhost:8000
echo  Health: http://localhost:8000/health
echo  Docs: http://localhost:8000/docs
echo ================================================
echo  Press Ctrl+C to stop
echo ================================================
echo.

python windows_voice_agent.py --host 0.0.0.0 --port 8000

echo.
echo Voice Agent stopped.
pause
