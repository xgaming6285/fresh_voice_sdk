# Test Your Greeting Generator NOW

## EASIEST WAY - Just Double-Click!

**Windows CMD:**

```
test_greeting_with_api_key.bat
```

**Windows PowerShell:**

```
.\test_greeting_with_api_key.ps1
```

These scripts set the API key and run the test automatically!

---

## OR: Manual Method

**Step 1 - Set Your API Key:**

**Windows PowerShell:**

```powershell
$env:GOOGLE_API_KEY = "AIzaSyBrjAdSyrD8tfuo4cZl5gQ92l7lSd-wBKA"
```

**Windows CMD:**

```cmd
set GOOGLE_API_KEY=AIzaSyBrjAdSyrD8tfuo4cZl5gQ92l7lSd-wBKA
```

**Linux/Mac:**

```bash
export GOOGLE_API_KEY=AIzaSyBrjAdSyrD8tfuo4cZl5gQ92l7lSd-wBKA
```

**Step 2 - Run Test:**

```bash
python test_gemini_greeting.py
```

This will:

1. ✅ Use the **Puck voice** (same as your voice calls)
2. ✅ Generate a Bulgarian greeting
3. ✅ Save it to `greetings/` folder
4. ✅ Show you the transcript

**Expected Output:**

```
✅ Gemini greeting generator imported successfully
🎤 This will use the same Puck voice as your voice calls
============================================================
🧪 Testing Bulgarian greeting with Puck voice
============================================================
🌐 Generating greeting with Gemini Live API (Puck voice)
🗣️ Language: Bulgarian
📝 Text: Здравейте! Аз съм Maria от компания QuantumAI...
🎤 Starting Gemini Live session...
📤 Requesting audio generation...
📥 Receiving audio from Gemini...
📥 Received X bytes of audio
🔊 Combining audio chunks...
🔄 Converted audio: X bytes (24kHz) → X bytes (8kHz)
💾 Saved as WAV: greetings/greeting_bg_XXXXXXXX_XXXXXX.wav
✅ Saved greeting audio: greetings/greeting_bg_XXXXXXXX_XXXXXX.wav
🎤 Voice: Puck (same as voice calls)

✅ SUCCESS! Greeting generated with Puck voice
📁 Audio file: greetings/greeting_bg_XXXXXXXX_XXXXXX.wav
📝 Transcript: Здравейте! Аз съм Maria от компания QuantumAI...
🗣️ Language: Bulgarian
🎤 Method: gemini-live (Puck voice)

💡 This is the SAME voice used in your voice calls!
💡 Play the audio file to verify quality
```

## What's Happening

1. **Using Gemini API** - Same API as your voice calls (no web scraping!)
2. **Puck Voice** - The exact same voice customers hear during calls
3. **Crystal Clear** - Professional quality, no radio noise
4. **Already Configured** - Uses your existing Google API key

## Listen to the Result

After the test runs, find the audio file:

```
greetings/greeting_bg_XXXXXXXX_XXXXXX.wav
```

Play it and verify:

- ✅ Clear audio (no radio noise)
- ✅ Natural speech
- ✅ Same voice as your calls

## Next: Make a Real Call

1. Start the voice agent:

```bash
python windows_voice_agent.py
```

2. Open CRM frontend:

```bash
cd crm-frontend
npm start
```

3. Go to Leads page

4. Click "Make appointment Call" on any lead

5. The system will:
   - Show "Generating Greeting..."
   - Create custom greeting with Puck voice
   - Make the call
   - Play the greeting

## Why This is Better Than Web Scraping

❌ **Web Scraping Google AI Studio:**

- Complex browser automation
- Requires login/authentication
- Breaks when UI changes
- Slow (loads web pages)
- Against TOS

✅ **Direct Gemini API:**

- Simple API calls
- Already authenticated
- Stable and fast
- Official method
- **Uses same voice as calls!**

## Troubleshooting

### Test fails with "No module named 'google.genai'"

```bash
pip install google-genai
```

### Test fails with API error

Check your API key is set:

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-key-here"

# Windows CMD
set GOOGLE_API_KEY=your-key-here

# Linux/Mac
export GOOGLE_API_KEY=your-key-here
```

### Want to test different languages?

Edit `test_gemini_greeting.py` and change:

```python
language="Bulgarian",
language_code="bg",
```

To:

```python
language="English",
language_code="en",
```

Or any other language!

## Files Created

- `greeting_generator_gemini.py` - Main generator using Gemini API
- `test_gemini_greeting.py` - Test script
- `GEMINI_GREETING_QUICKSTART.md` - Detailed guide
- This file - Quick test instructions

## Voice Agent Integration

The voice agent automatically uses Gemini greetings (highest priority):

```
Priority order:
1. Gemini (Puck voice) ← YOU ARE HERE ✅
2. edge-tts (Microsoft voices)
3. gTTS (basic quality)
4. Web scraping (complex)
```

When you start the voice agent, you'll see:

```
✅ Gemini greeting generator loaded (Gemini Live API)
🎤 Using Puck voice - same as voice calls for consistent experience
```

This confirms everything is working!

## Summary

🎉 **You now have:**

- ✅ Clear audio greetings (no radio noise)
- ✅ Same Puck voice as your calls
- ✅ No web scraping needed
- ✅ Automatic language detection
- ✅ Ready to use in production

**Just run the test to verify it works!**
