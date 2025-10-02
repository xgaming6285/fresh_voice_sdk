@echo off
echo ========================================
echo Testing Gemini Greeting Generator
echo ========================================
echo.

echo Setting API key...
set GOOGLE_API_KEY=AIzaSyBrjAdSyrD8tfuo4cZl5gQ92l7lSd-wBKA

echo.
echo Running test with Puck voice...
echo.
python test_gemini_greeting.py

echo.
echo ========================================
echo Test Complete!
echo ========================================
echo.
echo Check the greetings/ folder for the generated WAV file.
echo Play it to hear the Puck voice!
echo.
pause
