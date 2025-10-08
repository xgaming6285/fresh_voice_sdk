@echo off
echo =============================================
echo   SUBSCRIPTION SYSTEM - FRESH START
echo =============================================
echo.

echo [1/4] Stopping any running servers...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Voice Agent CRM Backend" >nul 2>&1
taskkill /f /im node.exe /fi "COMMANDLINE eq *react-scripts*" >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/4] Deleting old databases...
if exist voice_agent_crm.db (
    del voice_agent_crm.db
    echo   - voice_agent_crm.db deleted
)
if exist voice_agent_auth.db (
    del voice_agent_auth.db
    echo   - voice_agent_auth.db deleted
)

echo [3/4] Creating superadmin user...
python init_superadmin.py

echo [4/4] Starting servers...
start "Voice Agent CRM Backend" cmd /k "uvicorn windows_voice_agent:app --reload --port 8000"
timeout /t 3 /nobreak >nul
start "Voice Agent CRM Frontend" cmd /k "cd crm-frontend && npm start"

echo.
echo =============================================
echo   SETUP COMPLETE!
echo =============================================
echo.
echo IMPORTANT: Clear your browser storage:
echo   1. Open browser (http://localhost:3000)
echo   2. Press F12 to open DevTools
echo   3. Go to Console tab
echo   4. Run: localStorage.clear(); location.reload();
echo.
echo Login credentials:
echo   Superadmin: superadmin / 123123
echo.
echo Test Flow:
echo   1. Login as superadmin
echo   2. Create an admin (testadmin / 123123)
echo   3. Set payment wallet in Superadmin-^>Payments tab
echo   4. Logout, login as testadmin
echo   5. Try to create lead -^> Should show subscription error!
echo   6. Go to Billing page -^> Request agents
echo   7. Logout, login as superadmin
echo   8. Approve payment in Payments tab
echo   9. Login as testadmin -^> Now can create leads!
echo.
pause

