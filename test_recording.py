#!/usr/bin/env python3
"""
Test script for the CallRecorder functionality
Tests the recording of incoming and outgoing audio streams
"""

import os
import sys
import numpy as np
import wave
import json
from pathlib import Path
import time

# Add the current directory to path to import the recorder
sys.path.append('.')
from windows_voice_agent import CallRecorder

def generate_test_audio(duration_seconds=2, sample_rate=8000, frequency=440):
    """Generate test audio - a sine wave"""
    # Calculate number of samples
    num_samples = int(duration_seconds * sample_rate)
    
    # Generate time array
    t = np.linspace(0, duration_seconds, num_samples, False)
    
    # Generate sine wave (440 Hz A note)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_16bit = (sine_wave * 32767 * 0.5).astype(np.int16)  # 50% volume
    
    return audio_16bit.tobytes()

def test_call_recorder():
    """Test the CallRecorder class"""
    print("🧪 Testing CallRecorder functionality...")
    
    # Test session data
    session_id = "test-session-12345"
    caller_id = "+15551234567"
    called_number = "+15559876543"
    
    try:
        # Create recorder
        print(f"📞 Creating CallRecorder for session {session_id}")
        recorder = CallRecorder(session_id, caller_id, called_number)
        
        # Verify files were created
        print(f"📁 Session directory: {recorder.session_dir}")
        print(f"🎤 Incoming file: {recorder.incoming_wav_path}")
        print(f"🔊 Outgoing file: {recorder.outgoing_wav_path}")
        
        # Generate test audio
        print("🎵 Generating test audio...")
        incoming_audio = generate_test_audio(duration_seconds=3, frequency=440)  # A note
        outgoing_audio = generate_test_audio(duration_seconds=3, frequency=880)  # High A note
        
        # Record incoming audio (simulate multiple chunks)
        print("🎤 Recording incoming audio...")
        chunk_size = 320  # 20ms at 8kHz
        for i in range(0, len(incoming_audio), chunk_size):
            chunk = incoming_audio[i:i+chunk_size]
            recorder.record_incoming_audio(chunk)
            time.sleep(0.001)  # Small delay to simulate real-time
        
        # Record outgoing audio (simulate multiple chunks)
        print("🔊 Recording outgoing audio...")
        for i in range(0, len(outgoing_audio), chunk_size):
            chunk = outgoing_audio[i:i+chunk_size]
            recorder.record_outgoing_audio(chunk)
            time.sleep(0.001)  # Small delay to simulate real-time
        
        # Stop recording
        print("🛑 Stopping recording...")
        recorder.stop_recording()
        
        # Verify files exist and have content
        print("✅ Verifying recordings...")
        
        files_to_check = [
            ("Incoming", recorder.incoming_wav_path),
            ("Outgoing", recorder.outgoing_wav_path),
            ("Mixed", recorder.mixed_wav_path),
            ("Session Info", recorder.session_dir / "session_info.json")
        ]
        
        for file_type, file_path in files_to_check:
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✅ {file_type}: {file_path.name} ({size} bytes)")
                
                # Verify WAV files can be opened
                if file_path.suffix == '.wav':
                    try:
                        with wave.open(str(file_path), 'rb') as wav:
                            frames = wav.getnframes()
                            rate = wav.getframerate()
                            channels = wav.getnchannels()
                            duration = frames / rate
                            print(f"      Format: {rate}Hz, {channels}ch, {duration:.2f}s")
                    except Exception as e:
                        print(f"      ❌ Error reading WAV: {e}")
                
                # Verify JSON file can be parsed
                elif file_path.suffix == '.json':
                    try:
                        with open(file_path, 'r') as f:
                            session_info = json.load(f)
                        print(f"      Session ID: {session_info.get('session_id')}")
                        print(f"      Duration: {session_info.get('duration_seconds'):.2f}s")
                    except Exception as e:
                        print(f"      ❌ Error reading JSON: {e}")
            else:
                print(f"  ❌ {file_type}: {file_path.name} (missing)")
        
        print(f"✅ Test completed successfully!")
        print(f"📁 Files saved in: {recorder.session_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("🧹 Cleaning up test files...")
    sessions_dir = Path("sessions")
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir() and "test-session" in session_dir.name:
                print(f"🗑️  Removing {session_dir}")
                import shutil
                shutil.rmtree(session_dir)

if __name__ == "__main__":
    print("=" * 60)
    print("Call Recording Test")
    print("=" * 60)
    
    # Run test
    success = test_call_recorder()
    
    print("=" * 60)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
    
    # Ask if user wants to clean up
    response = input("\n🧹 Clean up test files? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        cleanup_test_files()
        print("✅ Test files cleaned up")
    else:
        print("📁 Test files preserved for inspection")
    
    print("=" * 60)
