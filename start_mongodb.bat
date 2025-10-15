@echo off
REM Start MongoDB Server
REM This script starts MongoDB as a Windows service

echo ============================================================
echo Starting MongoDB Server
echo ============================================================
echo.

REM Check if MongoDB service exists
sc query MongoDB >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] MongoDB service not found!
    echo.
    echo Please install MongoDB first. See MONGODB_SETUP.md for instructions.
    echo Download from: https://www.mongodb.com/try/download/community
    echo.
    pause
    exit /b 1
)

REM Check if MongoDB is already running
sc query MongoDB | find "RUNNING" >nul
if %errorlevel% equ 0 (
    echo [OK] MongoDB is already running!
) else (
    echo Starting MongoDB service...
    net start MongoDB
    if %errorlevel% equ 0 (
        echo [OK] MongoDB started successfully!
    ) else (
        echo [ERROR] Failed to start MongoDB!
        echo Try running this script as Administrator.
        pause
        exit /b 1
    )
)

echo.
echo MongoDB is running on: mongodb://localhost:27017/
echo Database name: voice_agent_crm
echo.
echo To test the connection, run: python test_mongodb_integration.py
echo.
pause

