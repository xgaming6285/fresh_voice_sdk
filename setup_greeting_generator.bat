@echo off
echo ========================================
echo Setting up High-Quality Greeting Generator
echo ========================================
echo.

echo Installing edge-tts for crystal-clear audio...
pip install edge-tts pydub

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo The HIGH-QUALITY greeting generator is now ready!
echo.
echo Features enabled:
echo - Microsoft Neural voices (no radio noise!)
echo - Automatic language detection from phone numbers
echo - Custom greeting generation in 30+ languages  
echo - Support for WAV and MP3 audio formats
echo - Male and female voice options
echo - No API key required!
echo.
echo Available voices include:
echo - Bulgarian: Boris (male), Kalina (female)
echo - Romanian: Emil (male), Alina (female)
echo - German: Conrad (male), Katja (female)
echo - And many more...
echo.
echo Alternative options (if edge-tts doesn't work):
echo   Simple TTS: pip install gTTS
echo   AI Studio: pip install playwright aiofiles
echo.
pause
