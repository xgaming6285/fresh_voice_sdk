# Call Recording Functionality

The Windows VoIP Voice Agent now includes comprehensive call recording functionality that automatically saves all voice calls as WAV audio files.

## Features

### 🎙️ Automatic Recording
- **Incoming Audio**: Records audio from the caller (what they say)
- **Outgoing Audio**: Records audio to the caller (what the AI says)  
- **Mixed Audio**: Creates a combined recording with both sides of the conversation
- **Session Metadata**: Saves call information in JSON format

### 📁 File Organization
Each call creates a dedicated session directory:
```
sessions/
└── {session-id}/
    ├── incoming_20241225_143022.wav    # Caller's audio
    ├── outgoing_20241225_143022.wav     # AI's audio
    ├── mixed_20241225_143022.wav        # Combined audio
    └── session_info.json                # Call metadata
```

### 🎵 Audio Format
- **Sample Rate**: 8 kHz (telephony quality)
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Format**: Uncompressed WAV files

## API Endpoints

### GET /api/recordings
Lists all available call recordings with metadata.

**Response Example:**
```json
{
  "status": "success",
  "total_recordings": 5,
  "recordings": [
    {
      "session_id": "abc123-def456-ghi789",
      "caller_id": "+15551234567",
      "called_number": "+15559876543", 
      "start_time": "2024-12-25T14:30:22.123456Z",
      "end_time": "2024-12-25T14:32:45.654321Z",
      "duration_seconds": 143.53,
      "audio_files": {
        "incoming_audio": {
          "filename": "incoming_20241225_143022.wav",
          "size_mb": 1.23,
          "path": "sessions/abc123-def456-ghi789/incoming_20241225_143022.wav"
        },
        "outgoing_audio": {
          "filename": "outgoing_20241225_143022.wav", 
          "size_mb": 2.14,
          "path": "sessions/abc123-def456-ghi789/outgoing_20241225_143022.wav"
        },
        "mixed_audio": {
          "filename": "mixed_20241225_143022.wav",
          "size_mb": 1.68,
          "path": "sessions/abc123-def456-ghi789/mixed_20241225_143022.wav"
        }
      }
    }
  ]
}
```

## Session Information

Each recording includes detailed metadata in `session_info.json`:

```json
{
  "session_id": "abc123-def456-ghi789",
  "caller_id": "+15551234567",
  "called_number": "+15559876543",
  "start_time": "2024-12-25T14:30:22.123456Z",
  "end_time": "2024-12-25T14:32:45.654321Z", 
  "duration_seconds": 143.53,
  "files": {
    "incoming_audio": "incoming_20241225_143022.wav",
    "outgoing_audio": "outgoing_20241225_143022.wav",
    "mixed_audio": "mixed_20241225_143022.wav"
  }
}
```

## Implementation Details

### CallRecorder Class
The `CallRecorder` class handles all recording functionality:

- **Automatic Initialization**: Created with each new call session
- **Real-time Recording**: Captures audio as it flows through the system
- **Memory Management**: Buffers are limited to prevent memory overflow  
- **Error Handling**: Graceful fallbacks if recording fails
- **Cleanup**: Automatically stops recording when calls end

### Integration Points

1. **RTPSession**: Records incoming audio from RTP packets
2. **WindowsVoiceSession**: Records outgoing audio to caller
3. **Session Cleanup**: Stops recording and generates mixed audio when calls end

### Audio Processing

- **Incoming**: Records decoded PCM audio from μ-law/A-law RTP packets
- **Outgoing**: Records PCM audio before μ-law encoding for transmission
- **Mixing**: Combines both streams by averaging samples to prevent clipping

## Testing

Use the included test script to verify recording functionality:

```bash
python test_recording.py
```

This creates a test session with synthetic audio and verifies:
- WAV files are created correctly
- Audio format is proper (8kHz, 16-bit, mono)
- Mixed audio combines both streams  
- Session metadata is generated
- File sizes are reasonable

## Storage Considerations

### Disk Space Usage
- **Telephony quality** (8kHz): ~1 MB per minute per stream
- **Typical call** (5 minutes): ~15 MB total (3 streams × 5 MB each)
- **Daily calls** (50 calls × 5 min): ~750 MB per day

### File Management
- Files are stored permanently until manually deleted
- Consider implementing automatic cleanup for old recordings
- Monitor disk space usage in production environments

## Troubleshooting

### Common Issues

1. **No audio recorded**: Check if session starts properly
2. **Empty files**: Verify audio is flowing through RTP
3. **Missing mixed file**: Check if both incoming and outgoing exist
4. **Permission errors**: Ensure write access to sessions directory

### Log Messages

Look for these log entries:
- `🎙️ Recording initialized for session {session_id}`
- `🛑 Stopping recording for session {session_id}`
- `📁 Incoming audio: {path} ({size} MB)`
- `🎵 Creating mixed audio recording...`

## Privacy and Legal Considerations

⚠️ **Important**: Call recording may be subject to legal requirements in your jurisdiction:

- **Consent**: Some regions require consent from all parties
- **Notification**: Callers may need to be informed of recording
- **Storage**: Consider encryption for sensitive conversations
- **Retention**: Implement policies for how long recordings are kept
- **Access**: Control who can access recorded conversations

Consult legal counsel to ensure compliance with applicable laws.
