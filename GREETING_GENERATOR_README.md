# Custom Greeting Generator for Voice Agent

This feature allows the CRM to generate custom greeting audio files in different languages before making outbound calls.

## Overview

When making an appointment call or any custom call from the CRM, the system will:

1. Detect the lead's language based on their phone number
2. Generate a custom greeting in that language with the configured company/product details
3. Use this custom greeting instead of the default `greeting.wav` file

## Installation

### Option 1: Gemini Live API - Puck Voice (BEST - Already Configured!)

**✨ Uses the SAME voice as your voice calls - perfect consistency!**

This option uses the Gemini Live API you already have configured for voice calls. It generates greetings with the **Puck voice** - the same voice your AI uses during calls.

```bash
# Already installed if you're running voice calls!
# Uses your existing Google API key
```

**Benefits:**

- ✅ Same Puck voice as your calls - perfect consistency
- ✅ Already configured - no additional setup
- ✅ Crystal-clear audio quality
- ✅ Supports all languages
- ✅ No extra API key needed

### Option 2: High-Quality TTS (Alternative - No API Key)

Install edge-tts for crystal-clear Microsoft Neural voices:

```bash
pip install edge-tts pydub
```

**✨ Benefits:**

- No radio noise or artifacts
- Professional neural voices
- 30+ languages with native accents
- Male and female voice options
- No API key or authentication required
- Free to use

### Option 3: Simple TTS (Basic Quality)

Install gTTS for basic text-to-speech:

```bash
pip install gTTS pydub
```

**⚠️ Note**: This may produce audio with some radio-like noise.

### Option 4: Google AI Studio Web Scraping (Not Recommended)

For Google AI Studio integration (requires manual setup):

```bash
pip install playwright aiofiles
playwright install chromium
```

**Note**: This is experimental and requires manual authentication.

## How It Works

1. **CRM Frontend**: When clicking "Make appointment Call" in the Custom Call Dialog:

   - The button shows "Generating Greeting..." while creating the audio
   - Calls the `/api/generate_greeting` endpoint
   - Waits for the greeting to be generated
   - Then makes the call with the custom greeting

2. **Backend API**: The `/api/generate_greeting` endpoint:

   - Detects the language from the phone number
   - Generates appropriate greeting text
   - Creates an audio file using TTS
   - Returns the file path

3. **Voice Agent**: When making the call:
   - Uses the custom greeting file instead of default `greeting.wav`
   - Supports both WAV and MP3 formats
   - Falls back to default greeting if custom file not found

## Supported Languages

The system automatically detects and generates greetings in:

- Bulgarian (bg)
- Romanian (ro)
- Greek (el)
- German (de)
- French (fr)
- Spanish (es)
- Italian (it)
- Russian (ru)
- English (en)
- And many more...

## Custom Greeting Format

The greeting text follows this pattern:

```
"Hello, I'm {caller_name} from {company_name}. Are you interested in {product_name}?"
```

This is automatically translated to the detected language.

## Configuration

In the Custom Call Dialog, you can configure:

- **Company Name**: The company you're representing
- **Caller Name**: The name of the AI caller
- **Product Name**: What you're selling
- **Additional options**: Call objective, urgency, benefits, etc.

## Fallback Behavior

If greeting generation fails:

- The system will use the default `greeting.wav` file
- The call will proceed normally
- An error will be logged but won't block the call

## API Endpoints

### Generate Greeting

```
POST /api/generate_greeting
{
  "phone_number": "+359888123456",
  "call_config": {
    "company_name": "QuantumAI",
    "caller_name": "Maria",
    "product_name": "ArtroFlex"
  }
}
```

Response:

```json
{
  "status": "success",
  "greeting_file": "greetings/greeting_bg_20241002_143022.wav",
  "transcript": "Здравейте, аз съм Maria от QuantumAI...",
  "language": "Bulgarian",
  "language_code": "bg"
}
```

### Make Call with Custom Greeting

```
POST /api/make_call
{
  "phone_number": "+359888123456",
  "call_config": { ... },
  "greeting_file": "greetings/greeting_bg_20241002_143022.wav"
}
```

## File Storage

Generated greeting files are stored in:

- `greetings/` directory
- Named with pattern: `greeting_{language}_{timestamp}.wav`
- Automatically cleaned up periodically (TODO)

## Troubleshooting

### "Greeting generator not available"

Install the high-quality dependencies:

```bash
pip install edge-tts pydub
```

### MP3 files not playing

Install pydub for MP3/WAV conversion:

```bash
pip install pydub
```

### Poor audio quality (radio noise)

You're likely using gTTS. Upgrade to edge-tts for crystal-clear audio:

```bash
pip install edge-tts
```

Then restart the voice agent. The system will automatically use the high-quality generator.

## Voice Quality Comparison

| Method       | Quality              | Noise       | Consistency      | API Key | Setup        |
| ------------ | -------------------- | ----------- | ---------------- | ------- | ------------ |
| Gemini Puck  | ⭐⭐⭐⭐⭐ Excellent | None        | ✅ Same as calls | Gemini  | ✅ Done      |
| edge-tts     | ⭐⭐⭐⭐⭐ Excellent | None        | Different voice  | No      | Easy         |
| gTTS         | ⭐⭐ Basic           | Radio noise | Different voice  | No      | Easy         |
| Web Scraping | ⭐⭐⭐⭐ Good        | None        | Complex          | Google  | Very Complex |

**🎤 Recommendation:** Use Gemini (Option 1) - it's already configured and uses the same Puck voice as your calls!

## Available Voices (edge-tts)

The high-quality generator includes native speakers for each language:

- **Bulgarian**: Boris (male), Kalina (female)
- **Romanian**: Emil (male), Alina (female)
- **Greek**: Nestoras (male), Athina (female)
- **German**: Conrad (male), Katja (female)
- **French**: Henri (male), Denise (female)
- **Spanish**: Alvaro (male), Elvira (female)
- **Italian**: Diego (male), Elsa (female)
- **Russian**: Dmitry (male), Svetlana (female)
- **English**: Christopher (male), Jenny (female)
- And many more...

## Future Improvements

1. **Voice Selection**: Allow choosing different AI voices
2. **Custom Scripts**: Support fully custom greeting scripts
3. **Caching**: Cache frequently used greetings
4. **Cleanup**: Automatic cleanup of old greeting files
5. **Preview**: Allow previewing greetings before calling
