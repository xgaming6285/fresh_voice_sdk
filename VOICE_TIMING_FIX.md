# Voice Agent Timing and Audio Quality Fixes

## Issues Addressed

1. **Agent waits for user to speak first** - The initial greeting wasn't being triggered
2. **Audio lag/choppiness** - Audio packets were being sent too fast without proper pacing

## Solutions Implemented

### 1. Fixed Initial Greeting Trigger

- Increased connection stabilization wait time to 0.5s
- Changed from pure silence to very quiet noise (1 second at low amplitude)
- Added explicit text instruction to greet the caller
- This ensures Gemini starts speaking immediately when the call connects

### 2. Fixed Audio Lag/Choppiness

- **Added proper RTP packet pacing**: Each 160-byte packet now waits 20ms before sending the next
- **Reduced audio buffer size**: Changed from 1600 bytes to 320 bytes for lower latency
- **More aggressive processing**: Process audio chunks as soon as they're available
- **Threaded packet sending**: RTP packets are sent in a separate thread with proper timing

## Technical Details

### RTP Packet Timing

- Each RTP packet contains 160 bytes of μ-law audio
- This represents 20ms of audio at 8kHz
- Packets must be sent with 20ms spacing to maintain proper playback speed
- Sending too fast causes audio buffer overflow and choppy playback

### Audio Processing Pipeline

1. Receive 160-byte RTP packets from Gate VoIP (8kHz μ-law)
2. Convert to 320-byte PCM chunks (16-bit)
3. Queue and process in 320-byte chunks (20ms)
4. Send to Gemini after upsampling to 16kHz
5. Receive response audio from Gemini (24kHz)
6. Downsample to 8kHz and convert to μ-law
7. Send back as properly-timed 160-byte RTP packets

## Testing the Fix

1. Call the agent
2. You should immediately hear: "Hello, how can I help you today?"
3. The audio should be clear and not choppy
4. Response should feel natural and conversational

## Debugging

If issues persist, check:

- Network latency between your system and Gate VoIP
- CPU usage during calls (high CPU can cause timing issues)
- Firewall/antivirus not interfering with UDP packets
- Gate VoIP codec settings (should be PCMU/G.711)
