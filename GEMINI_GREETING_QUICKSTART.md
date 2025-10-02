# Gemini Greeting Generator - Quick Start

## Why Use Gemini for Greetings?

✅ **Same Voice** - Uses the Puck voice, identical to your voice calls  
✅ **Already Configured** - Uses your existing Gemini API key  
✅ **Crystal Clear** - Professional quality audio, no noise  
✅ **No Extra Setup** - Works immediately if voice calls work

## How It Works

Instead of web scraping Google AI Studio, this generator uses the **Gemini Live API directly** - the same API your voice agent uses for calls. This means:

1. **No complex web automation** - Direct API calls
2. **Same authentication** - Uses your existing API key
3. **Puck voice** - The same voice customers hear during calls
4. **Consistent experience** - Greeting sounds exactly like the AI in calls

## Test It Now

```bash
python test_gemini_greeting.py
```

This will:

- Generate a Bulgarian greeting with Puck voice
- Save it as a WAV file in the `greetings/` folder
- Show you the transcript

## What Happens Behind the Scenes

1. **Text Generation**: Creates greeting text in the target language
2. **Gemini Live API**: Sends text to Gemini with Puck voice config
3. **Audio Streaming**: Receives high-quality audio from Gemini
4. **Format Conversion**: Converts from 24kHz to 8kHz telephony format
5. **WAV File**: Saves as standard WAV file for playback

## Configuration

No configuration needed! The system uses:

- Your existing `GOOGLE_API_KEY` environment variable
- The same Gemini client as voice calls
- The same voice config (Puck voice)

## When Greeting Is Generated

The greeting is generated **before each call** when you:

1. Click "Make appointment Call" in the CRM
2. The button shows "Generating Greeting..."
3. System detects language from phone number
4. Gemini generates audio with Puck voice
5. Call starts with the custom greeting

## Voice Consistency

This is the **best option** because:

```
Call Flow:
1. Greeting plays → [Puck voice from greeting generator]
2. User responds
3. AI responds → [Puck voice from live API]

Same voice throughout = Professional & consistent!
```

With other TTS methods:

```
Call Flow:
1. Greeting plays → [Different voice: Microsoft/Google TTS]
2. User responds
3. AI responds → [Puck voice from Gemini]

Different voices = Inconsistent experience ❌
```

## Language Support

Puck voice supports all the languages Gemini supports:

- Bulgarian (Български)
- Romanian (Română)
- Greek (Ελληνικά)
- German (Deutsch)
- French (Français)
- Spanish (Español)
- Italian (Italiano)
- Russian (Русский)
- English
- And many more...

## Example Generated Greeting

**Bulgarian:**

> "Здравейте! Аз съм Maria от компания QuantumAI. Обаждам се във връзка с АртроФлекс. Интересувате ли се да научите повече?"

**Audio:** Crystal-clear Puck voice, same as in calls

## Troubleshooting

### "No module named 'google.genai'"

Your voice calls won't work either. Install:

```bash
pip install google-genai
```

### "Failed to generate greeting audio"

Check:

1. Is `GOOGLE_API_KEY` environment variable set?
2. Do voice calls work? (Same API key)
3. Check the logs for specific error messages

### "Audio sounds different from calls"

You might be using a different greeting generator. Check logs:

```
✅ Gemini greeting generator loaded (Gemini Live API)
🎤 Using Puck voice - same as voice calls
```

If you see something else (edge-tts, gTTS), the Gemini generator isn't loading.

## Cost

Uses the same Gemini API quota as your voice calls. A typical greeting:

- ~5 seconds of audio
- Minimal API cost
- Same billing as voice calls

## Files Created

Greetings are saved in:

```
greetings/
  ├── greeting_bg_20241002_143022.wav
  ├── greeting_ro_20241002_143145.wav
  └── greeting_en_20241002_143230.wav
```

Format: `greeting_{language_code}_{timestamp}.wav`

## Next Steps

1. **Test it**: `python test_gemini_greeting.py`
2. **Make a call**: Use the CRM to make a custom call
3. **Listen**: Verify the greeting uses Puck voice
4. **Enjoy**: Consistent voice experience throughout calls!

## Why Not Web Scraping?

You might wonder why we don't scrape Google AI Studio web interface:

**Web Scraping Issues:**

- ❌ Requires browser automation (Playwright)
- ❌ Needs authentication handling
- ❌ Breaks when UI changes
- ❌ Slow (waits for page loads)
- ❌ Complex error handling
- ❌ May violate TOS

**Direct API Approach:**

- ✅ Simple HTTP requests
- ✅ Already authenticated
- ✅ Stable API contract
- ✅ Fast response times
- ✅ Clean error messages
- ✅ Official supported method

**Result**: Same Puck voice, much simpler implementation!
