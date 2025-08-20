# Voice Assistant with Session Logging

A voice assistant powered by Google Gemini Live API with comprehensive session logging capabilities.

## Features

- 🎤 Real-time voice interaction with Google Gemini
- 📝 Automatic transcript logging to MongoDB
- 🎵 Audio recording (WAV + MP3 format)
- 📊 Session metadata tracking
- 💾 Fallback to local file storage if MongoDB is unavailable

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. MongoDB Setup (Optional)

- Install MongoDB locally and run on port 27017 (default)
- Or skip this step - the app will work without MongoDB and save files locally

### 3. Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

### 4. Run the Application

```bash
python main.py
```

Options:

- `--mode camera` (default): Share camera feed
- `--mode screen`: Share screen capture
- `--mode none`: Audio only

## Session Logging

### What Gets Logged

1. **Transcripts** (if available):

   - User speech (when model generates text)
   - Assistant responses
   - Timestamps for all interactions

2. **Audio Files**:

   - Input audio (your voice) as WAV and MP3
   - Output audio (assistant voice) as WAV and MP3

3. **Session Metadata**:
   - Session ID and duration
   - Message counts
   - Audio file availability

### Storage Locations

#### MongoDB (Primary)

- Database: `voice_assistant`
- Collection: `sessions`
- Connection: `localhost:27017`

#### Local Files (Backup/Fallback)

```
sessions/
├── [session-id]/
│   ├── input_[session-id].wav
│   ├── input_[session-id].mp3
│   ├── output_[session-id].wav
│   ├── output_[session-id].mp3
│   └── transcript_[session-id].json (if MongoDB unavailable)
```

### MongoDB Document Structure

```json
{
  "session_id": "uuid-string",
  "start_time": "ISO-datetime",
  "end_time": "ISO-datetime",
  "transcript": [
    {
      "timestamp": "ISO-datetime",
      "type": "user_input|assistant_response",
      "content": "text content"
    }
  ],
  "metadata": {
    "total_messages": 10,
    "session_duration": 120.5,
    "audio_files_saved": {
      "input": true,
      "output": true
    }
  }
}
```

## Usage

1. **Start the application**: `python main.py`
2. **Speak naturally** - the assistant will respond
3. **Check logs**:
   - MongoDB: Connect to see real-time sessions
   - Files: Check `sessions/` directory
4. **Exit**: Press `Ctrl+C` to stop and save session

## Troubleshooting

### MongoDB Connection Issues

- App continues working without MongoDB
- Sessions save as local JSON + audio files
- Check MongoDB is running: `mongod --version`

### Audio Issues

- Use headphones to prevent feedback
- Check microphone permissions
- Try different `--mode` options

### Missing Dependencies

```bash
# For MongoDB
pip install pymongo

# For audio processing
pip install pydub

# For audio playback/recording
pip install pyaudio
```

## Session Data

Each session creates:

- ✅ Unique session ID
- ✅ Complete audio recordings (input/output)
- ✅ Text transcripts (when available)
- ✅ Timestamps and metadata
- ✅ MongoDB storage (primary) + file backup

Perfect for conversation analysis, voice data collection, and session review!
