@echo off
REM Simple batch script to test outbound call with SIP code monitoring

echo.
echo ========================================
echo   TEST OUTBOUND CALL - SIP Monitor
echo ========================================
echo.
echo Calling: +359988925337
echo Gate Slot: 9
echo.
echo Watch for these SIP codes in the logs:
echo   [100 Trying]  - Call initiated
echo   [180 Ringing] - Phone is ringing ^<-- TARGET!
echo   [200 OK]      - Call answered
echo.
echo ========================================
echo.

python test_outbound_call.py

pause

