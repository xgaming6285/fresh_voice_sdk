# Audio Noise Fix for Windows Voice Agent

## Problem

The voice agent was producing noise similar to radio frequency interference during audio playback. This was caused by several issues in the audio processing pipeline.

## Root Causes Identified

1. **Sample Rate Conversion Artifacts**

   - Harsh downsampling from 24kHz to 8kHz was causing aliasing
   - Multiple cascaded filters were introducing phase distortion

2. **Audio Discontinuities**

   - Large audio chunks were being sent without proper buffering
   - No fade-in/fade-out causing clicks and pops between chunks

3. **μ-law Conversion Issues**

   - Direct conversion without headroom was causing clipping artifacts
   - No soft limiting before compression

4. **RTP Stream Gaps**
   - Silence gaps in the RTP stream were causing harsh transitions
   - No comfort noise generation during pauses

## Fixes Applied

### 1. Improved Audio Conversion (`convert_gemini_to_telephony`)

- **Gentler Anti-aliasing**: Reduced cutoff frequency from 3800Hz to 3200Hz for cleaner audio
- **Single Band-pass Filter**: Replaced cascaded filters with single band-pass to reduce phase distortion
- **Soft Limiting**: Applied gentle compression instead of harsh normalization
- **Fade-in/Fade-out**: Added 1ms fades to prevent clicks at chunk boundaries
- **Removed DC Offset Removal**: This was causing artifacts

### 2. Better Audio Buffering (`send_audio`)

- **Chunk Splitting**: Large audio chunks are now split into 40ms pieces for smoother delivery
- **Small Chunk Buffering**: Audio fragments smaller than 20ms are buffered to avoid fragmentation

### 3. Enhanced μ-law Conversion (`pcm_to_ulaw`)

- **Soft Limiting**: Apply 95% headroom before conversion to prevent clipping
- **Proper Sample Handling**: Convert to sample array for precise limiting
- **Error Handling**: Fallback to simple conversion if advanced processing fails

### 4. RTP Stream Continuity (`_process_output_queue`)

- **Comfort Noise Generation**: Generate low-level comfort noise during silence gaps
- **Silence Counter**: Limit comfort noise to 100ms to avoid excessive padding
- **Better Queue Management**: Improved timeout handling for responsiveness

### 5. Consistent Audio Buffering

- Applied buffering logic to all audio receiving paths
- Prevents small fragments from causing discontinuities
- Maintains minimum 20ms chunks for stable playback

## Results

These changes should eliminate the radio-like noise and provide:

- Cleaner audio output
- Smoother transitions between audio chunks
- Better handling of silence periods
- Reduced clipping and distortion
- More stable RTP stream delivery

## Testing

To verify the fixes:

1. Make a test call to the voice agent
2. Listen for any noise, clicks, or distortion
3. Pay attention to transitions between speech segments
4. Check audio quality during pauses in conversation
