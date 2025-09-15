# RTP Audio Pacing Fix

## Problem

Audio was lagging and choppy because RTP packets were being sent too fast without proper timing. When Gemini sent large chunks of audio (e.g., 11520 bytes), all packets were sent immediately, overwhelming the receiver.

## Solution

Implemented a proper RTP pacing mechanism using an output queue:

1. **Output Queue System**:

   - Audio from Gemini is queued instead of sent immediately
   - A dedicated thread processes the queue with proper timing

2. **Precise Packet Timing**:

   - Each RTP packet represents 20ms of audio (160 bytes of μ-law)
   - Packets are sent exactly 20ms apart using high-precision timing
   - This matches the natural playback rate of the audio

3. **Benefits**:
   - Smooth, consistent audio playback
   - No buffer overruns or underruns
   - Proper synchronization with telephony systems

## Technical Details

- RTP packets: 160 bytes (20ms of μ-law audio at 8kHz)
- Packet interval: 20ms (50 packets per second)
- Output queue: Decouples audio reception from transmission
- Timing: Uses time.sleep() with compensation for processing delays

This ensures that audio is delivered at the correct rate regardless of how fast Gemini sends response chunks.
