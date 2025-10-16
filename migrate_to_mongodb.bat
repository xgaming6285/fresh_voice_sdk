@echo off
echo ============================================
echo MongoDB Migration Script
echo ============================================
echo.
echo This script will migrate your data from SQLite to MongoDB
echo.
echo Prerequisites:
echo   - MongoDB must be running on localhost:27017
echo   - Existing voice_agent_crm.db file (if you have data to migrate)
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Starting migration...
python migrate_sqlite_to_mongodb.py

echo.
echo Migration complete!
echo.
pause

