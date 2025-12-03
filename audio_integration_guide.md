# Audio Processing Integration Guide

## Current Issues in windows_voice_agent.py

### 1. Poor Resampling Quality
**Current Code (Lines 3901-3940):**
```python
# Simple upsampling - introduces aliasing
out_array = np.repeat(in_array, 2)

# Basic decimation - can cause artifacts  
out_array = decimate(in_array, 3, ftype='iir', zero_phase=True)
```

**Problems:**
- `np.repeat()` creates aliasing artifacts (radio-like noise)
- `decimate()` with IIR filter can introduce ringing
- No proper anti-aliasing for upsampling

### 2. G.711 Codec Issues
**Current Code (Lines 2213-2236):**
```python
def ulaw_to_pcm(self, ulaw_data: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw_data, 2)

def pcm_to_ulaw(self, pcm_data: bytes) -> bytes:
    return audioop.lin2ulaw(pcm_data, 2)
```

**Problems:**
- `audioop` is slow and basic
- Multiple conversions degrade quality
- No optimization for real-time processing

### 3. Latency Issues
**Current Code (Lines 1815-1820):**
```python
# 40ms buffering
min_chunk = 640  # 40ms at 8kHz
if len(self.audio_buffer) >= min_chunk:
    chunk_to_send = self.audio_buffer
```

**Problems:**
- 40ms buffering + 20ms RTP = 60ms+ latency
- Fixed buffering doesn't adapt to network conditions
- No jitter buffer management

## Recommended Solution

### 1. Install Required Libraries
```bash
pip install resampy librosa soundfile
```

### 2. Replace Audio Processing Methods

**In WindowsVoiceSession class, replace these methods:**

```python
def convert_telephony_to_gemini(self, audio_data: bytes) -> bytes:
    """Replace with optimized version"""
    if not hasattr(self, '_audio_processor'):
        from optimized_audio_processor import OptimizedAudioProcessor
        self._audio_processor = OptimizedAudioProcessor(
            input_sample_rate=8000,
            output_sample_rate=16000,
            chunk_size_ms=20,
            enable_noise_reduction=True,
            enable_agc=True
        )
    
    return self._audio_processor.process_chunk(audio_data)

def convert_gemini_to_telephony(self, model_pcm: bytes) -> bytes:
    """Replace with optimized version"""
    if not hasattr(self, '_audio_processor_down'):
        from optimized_audio_processor import OptimizedAudioProcessor
        self._audio_processor_down = OptimizedAudioProcessor(
            input_sample_rate=24000,
            output_sample_rate=8000,
            chunk_size_ms=20,
            enable_noise_reduction=False,  # Don't process AI output
            enable_agc=False
        )
    
    return self._audio_processor_down.process_chunk(model_pcm)
```

### 3. Replace G.711 Codec Methods

**In RTPSession class, replace these methods:**

```python
def ulaw_to_pcm(self, ulaw_data: bytes) -> bytes:
    """Replace with optimized version"""
    if not hasattr(self, '_g711_optimizer'):
        from optimized_audio_processor import G711Optimizer
        self._g711_optimizer = G711Optimizer()
    
    # Convert to float32 PCM
    pcm_float = self._g711_optimizer.decode_ulaw_fast(ulaw_data)
    # Convert to 16-bit bytes
    return (pcm_float * 32767.0).astype(np.int16).tobytes()

def pcm_to_ulaw(self, pcm_data: bytes) -> bytes:
    """Replace with optimized version"""
    if not hasattr(self, '_g711_optimizer'):
        from optimized_audio_processor import G711Optimizer
        self._g711_optimizer = G711Optimizer()
    
    # Convert bytes to float32
    pcm_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    return self._g711_optimizer.encode_ulaw_fast(pcm_array)
```

### 4. Optimize Buffering Strategy

**Replace the buffering logic (Lines 1811-1821):**

```python
# Reduce buffer size for lower latency
min_chunk = 320  # 20ms at 8kHz = 160 samples * 2 bytes = 320 bytes

# Adaptive buffering based on processing time
if hasattr(self, '_audio_processor'):
    stats = self._audio_processor.get_performance_stats()
    if stats.get('status') == 'ok':
        processing_time = stats.get('avg_processing_time_ms', 0)
        # Adjust buffer size based on processing performance
        if processing_time < 5:  # Very fast processing
            min_chunk = 160  # 10ms chunks
        elif processing_time > 15:  # Slower processing
            min_chunk = 480  # 30ms chunks
```

### 5. Add Performance Monitoring

**Add this method to WindowsVoiceSession:**

```python
def get_audio_performance_stats(self) -> dict:
    """Get audio processing performance statistics"""
    stats = {}
    
    if hasattr(self, '_audio_processor'):
        stats['upsampling'] = self._audio_processor.get_performance_stats()
    
    if hasattr(self, '_audio_processor_down'):
        stats['downsampling'] = self._audio_processor_down.get_performance_stats()
    
    if hasattr(self, '_g711_optimizer'):
        stats['g711_available'] = True
    
    return stats
```

## Expected Improvements

### 1. Audio Quality
- **Eliminated aliasing**: High-quality resampling with proper anti-aliasing
- **Reduced artifacts**: No more radio-like noise from poor upsampling
- **Better codec**: Optimized G.711 with pre-computed tables

### 2. Latency Reduction
- **20ms chunks**: Reduced from 40ms buffering
- **Adaptive buffering**: Adjusts based on processing performance
- **Faster processing**: Optimized algorithms and lookup tables

### 3. Performance Monitoring
- **Real-time stats**: Monitor processing times
- **Adaptive optimization**: Automatically adjust parameters
- **Quality metrics**: Track audio processing efficiency

## Testing the Changes

### 1. Performance Test
```python
# Add to health check endpoint
@app.get("/api/audio/performance")
async def get_audio_performance():
    performance_data = {}
    
    # Test with active sessions
    for session_id, session_data in active_sessions.items():
        voice_session = session_data["voice_session"]
        if hasattr(voice_session, 'get_audio_performance_stats'):
            performance_data[session_id] = voice_session.get_audio_performance_stats()
    
    return {
        "status": "success",
        "sessions": performance_data,
        "optimization_status": "enabled" if performance_data else "no_active_sessions"
    }
```

### 2. Quality Test
```python
# Add audio quality metrics
def measure_audio_quality(original: bytes, processed: bytes) -> dict:
    """Measure audio quality metrics"""
    orig_array = np.frombuffer(original, dtype=np.int16).astype(np.float32)
    proc_array = np.frombuffer(processed, dtype=np.int16).astype(np.float32)
    
    # Signal-to-noise ratio
    signal_power = np.mean(orig_array ** 2)
    noise_power = np.mean((orig_array - proc_array) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Total harmonic distortion
    # ... (implementation details)
    
    return {
        "snr_db": snr,
        "processing_artifacts": "minimal" if snr > 40 else "moderate" if snr > 20 else "high"
    }
```

## Migration Steps

1. **Install dependencies**: `pip install resampy librosa soundfile`
2. **Add optimized_audio_processor.py** to your project
3. **Test in development** with a single call
4. **Monitor performance** using the new metrics
5. **Gradually roll out** to production calls
6. **Fine-tune parameters** based on real-world performance

## Configuration Options

```python
# For high-quality, low-latency (recommended)
processor = OptimizedAudioProcessor(
    input_sample_rate=8000,
    output_sample_rate=16000,
    chunk_size_ms=20,
    enable_noise_reduction=True,
    enable_agc=True,
    enable_echo_suppression=False  # CPU intensive
)

# For maximum performance (minimal processing)
processor = OptimizedAudioProcessor(
    input_sample_rate=8000,
    output_sample_rate=16000,
    chunk_size_ms=10,  # Even lower latency
    enable_noise_reduction=False,
    enable_agc=False,
    enable_echo_suppression=False
)

# For maximum quality (higher CPU usage)
processor = OptimizedAudioProcessor(
    input_sample_rate=8000,
    output_sample_rate=16000,
    chunk_size_ms=20,
    enable_noise_reduction=True,
    enable_agc=True,
    enable_echo_suppression=True
)
```

This solution should eliminate the lag and radio-like noise while providing better overall audio quality and performance monitoring capabilities.