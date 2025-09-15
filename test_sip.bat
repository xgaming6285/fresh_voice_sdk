@echo off
REM SIP Connectivity Test Script

echo ================================================
echo  SIP Connectivity Test for Gate VoIP
echo ================================================
echo  This will test if Gate VoIP can reach your PC
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [INFO] Starting SIP test listener...
echo [INFO] This will listen for SIP messages on port 5060
echo.
echo INSTRUCTIONS:
echo 1. Leave this window open
echo 2. Call +359898995151 from any phone
echo 3. Watch for SIP messages in this window
echo 4. Press Ctrl+C to stop
echo.
echo ================================================

python test_sip_connectivity.py

echo.
echo [INFO] SIP test completed
pause
