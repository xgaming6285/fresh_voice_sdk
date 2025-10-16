@echo off
echo ========================================
echo Session Data Migration to MongoDB
echo ========================================
echo.

python migrate_sessions_to_mongodb.py

echo.
echo Press any key to exit...
pause >nul

