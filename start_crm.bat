@echo off
echo Starting Voice Agent CRM System...
echo.

REM Start backend in new window
echo Starting Voice Agent Backend...
start "Voice Agent Backend" cmd /k "python windows_voice_agent.py"

REM Wait a bit for backend to start
timeout /t 5 /nobreak > nul

REM Start frontend in new window
echo Starting CRM Frontend...
cd crm-frontend
start "CRM Frontend" cmd /k "npm start"

echo.
echo ============================================
echo Voice Agent CRM is starting up...
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo.
echo The browser will open automatically when ready.
echo ============================================
echo.
echo Press any key to exit this window...
pause > nul
