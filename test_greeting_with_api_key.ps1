Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Gemini Greeting Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Setting API key..." -ForegroundColor Yellow
$env:GOOGLE_API_KEY = "AIzaSyBrjAdSyrD8tfuo4cZl5gQ92l7lSd-wBKA"

Write-Host ""
Write-Host "Running test with Puck voice..." -ForegroundColor Green
Write-Host ""
python test_gemini_greeting.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check the greetings/ folder for the generated WAV file." -ForegroundColor Yellow
Write-Host "Play it to hear the Puck voice!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
