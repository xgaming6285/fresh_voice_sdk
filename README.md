# Voice Assistant with Session Logging & Telephony Integration

A comprehensive voice assistant powered by Google Gemini Live API with session logging and full telephony integration capabilities.

## Features

### Core Voice Assistant

- 🎤 Real-time voice interaction with Google Gemini
- 📝 Automatic transcript logging to MongoDB
- 🎵 Audio recording (WAV + MP3 format)
- 📊 Session metadata tracking
- 💾 Fallback to local file storage if MongoDB is unavailable

### 📞 NEW: Telephony Integration

- ☎️ **Incoming call handling** via SIM gateway/Asterisk
- 📞 **Outgoing call capabilities** through REST API
- 🏢 **Enterprise-grade telephony** with SIP trunk support
- 🔄 **Real-time audio conversion** between telephony and AI formats
- 📋 **DTMF support** (keypad input during calls)
- 👥 **Call transfer** to human operators
- 📊 **Call analytics** and session tracking

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. MongoDB Setup

**Option A: MongoDB Atlas (Cloud) - Recommended**

Create a `.env` file in the project root:

```env
# MongoDB Atlas connection string
MONGO_DB=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=voice_agent_crm
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

See [MONGODB_SETUP.md](MONGODB_SETUP.md) for detailed setup instructions.

**Option B: Local MongoDB**

- Install MongoDB locally and run on port 27017 (default)
- Or skip this step - the app will work without MongoDB and save files locally

Create a `.env` file:

```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=voice_agent_crm
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

### 3. Test MongoDB Connection

```bash
python test_mongodb_connection.py
```

### 4. Run the Application

```bash
python main.py
```

Options:

- `--mode camera` (default): Share camera feed
- `--mode screen`: Share screen capture
- `--mode none`: Audio only

## 📞 Telephony Integration Setup

### Prerequisites for Phone Integration

- **Asterisk server** (for SIP/telephony handling)
- **SIM gateway** or VoIP provider access
- **Linux environment**:
  - **Linux server** (Ubuntu/CentOS recommended), OR
  - **Windows 11 + WSL** (Windows Subsystem for Linux) ✅
  - **Docker** (alternative option)
- **Network access** between components

### Quick Telephony Setup

1. **Run the automated setup**:

   ```bash
   # On Linux systems or WSL
   bash setup_telephony.sh

   # Windows 11 users: Install WSL first
   # In PowerShell (as Administrator): wsl --install
   # Then follow SETUP_WINDOWS_WSL.md
   ```

2. **Configure your SIM gateway**:

   - Update `asterisk_config.json` with your gateway details
   - Configure SIP trunk to point to your Asterisk server
   - Set up call routing in your VoIP panel

3. **Start the telephony voice server**:

   ```bash
   python agi_voice_server.py --host 0.0.0.0 --port 8000
   ```

4. **Test the integration**:

   ```bash
   # Run comprehensive tests
   python test_telephony.py

   # Make a test call via API
   curl -X POST http://localhost:8000/api/make_call \
     -H "Content-Type: application/json" \
     -d '{"phone_number": "+1234567890"}'
   ```

### 📖 Detailed Setup Guide

For complete telephony integration instructions, see:

- **`SETUP_TELEPHONY.md`** - Step-by-step Linux configuration guide
- **`SETUP_WINDOWS_WSL.md`** - Windows 11 + WSL setup guide 🖥️
- **`INTEGRATION_SUMMARY.md`** - Architecture overview and features

### Telephony API Endpoints

Once running, these endpoints are available:

- `GET /health` - System health check
- `GET /api/sessions` - View active calls
- `POST /api/make_call` - Initiate outbound calls
- `WS /ws/audio/{session_id}` - Real-time audio streaming

### 🔗 Call Tracking with Asterisk Linked ID

The system captures Asterisk's `linkedid` for call-wide tracking:

- **What is linkedid?** A unique identifier that remains consistent across all legs of a call (A-leg, B-leg, transfers, etc.)
- **Why use linkedid?** Unlike `uniqueid` which is per-channel, `linkedid` tracks the entire call session across transfers, making it ideal for CDR (Call Detail Records) tracking
- **Where to find it?** The linkedid is printed in the terminal logs when making/receiving calls:
  ```
  🔗 Asterisk Linked ID (Call-Wide): 1636547890.1
  ```

**Enabling Linked ID in Asterisk:**

1. Update your Asterisk dialplan (see `asterisk_dialplan.conf` for full config):
   ```
   exten => _X.,n,Set(LINKEDID=${CHANNEL(linkedid)})
   exten => _X.,n,Set(PJSIP_HEADER(add,X-Asterisk-Linkedid)=${LINKEDID})
   ```

2. Reload Asterisk dialplan:
   ```bash
   asterisk -rx "dialplan reload"
   ```

3. The voice agent will automatically capture and display the linkedid in logs

**Testing:**
Run `python test_outbound_call.py` and check the logs for the linkedid output.

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
