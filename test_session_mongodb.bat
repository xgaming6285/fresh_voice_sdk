@echo off
echo ========================================
echo Test Session MongoDB Integration
echo ========================================
echo.

echo Running ORM access test...
python test_session_mongodb_api.py
echo.

echo Running API response verification...
python verify_session_api_response.py
echo.

echo ========================================
echo All tests complete!
echo ========================================
echo.
echo Press any key to exit...
pause >nul

