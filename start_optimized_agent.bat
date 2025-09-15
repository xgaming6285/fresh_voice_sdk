@echo off
echo ========================================
echo Starting Optimized Voice Agent
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
echo Checking dependencies...
pip show google-generativeai >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_windows.txt
)

REM Check for API key
if "%GEMINI_API_KEY%"=="" (
    echo.
    echo ERROR: GEMINI_API_KEY environment variable not set
    echo Please set it using: set GEMINI_API_KEY=your-api-key-here
    echo.
    pause
    exit /b 1
)

REM Start the optimized agent
echo.
echo Starting optimized voice agent...
echo This version has reduced latency for faster responses
echo.
python windows_voice_agent_optimized.py --port 8001

pause
