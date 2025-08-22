# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install -r requirements.txt
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python main.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python main.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import wave
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import cv2
import pyaudio
import PIL.Image
import mss
from pymongo import MongoClient
from pydub import AudioSegment

import argparse

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1beta"})

CONFIG = {
    "response_modalities": ["AUDIO"], 
    "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}}
}

pya = pyaudio.PyAudio()


class SessionLogger:
    def __init__(self, mongo_host="localhost", mongo_port=27017, db_name="voice_assistant", collection_name="sessions"):
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now(timezone.utc)
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.transcript_log = []
        self.audio_data = {
            "input": [],
            "output": []
        }
        
        # Audio recording parameters
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(exist_ok=True)
        self.session_dir = self.sessions_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Try to connect to MongoDB
        try:
            self.mongo_client = MongoClient(mongo_host, mongo_port, serverSelectionTimeoutMS=5000)
            # Test connection
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[db_name]
            self.collection = self.db[collection_name]
            print(f"✅ Connected to MongoDB at {mongo_host}:{mongo_port}")
            print(f"📝 Session ID: {self.session_id}")
        except Exception as e:
            print(f"⚠️  MongoDB connection failed: {e}")
            print("📁 Will save audio files only")
            self.mongo_client = None
    
    def log_transcript(self, message_type, content, timestamp=None):
        """Log transcript messages (user input or assistant response)"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        log_entry = {
            "timestamp": timestamp,
            "type": message_type,  # "user_input" or "assistant_response"
            "content": content
        }
        
        self.transcript_log.append(log_entry)
        print(f"📝 [{message_type}]: {content}")
    
    def save_audio_chunk(self, audio_data, audio_type, chunk_index):
        """Save individual audio chunks for later processing"""
        self.audio_data[audio_type].append({
            "data": audio_data,
            "timestamp": datetime.now(timezone.utc),
            "chunk_index": chunk_index
        })
    
    def save_audio_files(self):
        """Convert and save audio data as WAV and MP3 files"""
        try:
            for audio_type in ["input", "output"]:
                if not self.audio_data[audio_type]:
                    continue
                
                # Combine all audio chunks
                combined_audio = b""
                for chunk in self.audio_data[audio_type]:
                    combined_audio += chunk["data"]
                
                if not combined_audio:
                    continue
                
                # Save as WAV
                wav_file = self.session_dir / f"{audio_type}_{self.session_id}.wav"
                
                # Determine sample rate based on audio type
                sample_rate = SEND_SAMPLE_RATE if audio_type == "input" else RECEIVE_SAMPLE_RATE
                
                with wave.open(str(wav_file), 'wb') as wav_writer:
                    wav_writer.setnchannels(CHANNELS)
                    wav_writer.setsampwidth(pya.get_sample_size(FORMAT))
                    wav_writer.setframerate(sample_rate)
                    wav_writer.writeframes(combined_audio)
                
                print(f"💾 Saved {audio_type} audio: {wav_file}")
                
                # Convert to MP3
                try:
                    audio_segment = AudioSegment.from_wav(str(wav_file))
                    mp3_file = self.session_dir / f"{audio_type}_{self.session_id}.mp3"
                    audio_segment.export(str(mp3_file), format="mp3")
                    print(f"💾 Saved {audio_type} audio: {mp3_file}")
                except Exception as e:
                    print(f"⚠️  Could not convert {audio_type} to MP3: {e}")
                    
        except Exception as e:
            print(f"❌ Error saving audio files: {e}")
    
    def save_session_to_mongodb(self):
        """Save the complete session to MongoDB"""
        if not self.mongo_client:
            return False
        
        try:
            session_doc = {
                "session_id": self.session_id,
                "start_time": self.session_start,
                "end_time": datetime.now(timezone.utc),
                "transcript": self.transcript_log,
                "metadata": {
                    "total_messages": len(self.transcript_log),
                    "session_duration": (datetime.now(timezone.utc) - self.session_start).total_seconds(),
                    "audio_files_saved": {
                        "input": len(self.audio_data["input"]) > 0,
                        "output": len(self.audio_data["output"]) > 0
                    }
                }
            }
            
            result = self.collection.insert_one(session_doc)
            print(f"✅ Session saved to MongoDB with ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving to MongoDB: {e}")
            return False
    
    def save_session(self):
        """Save session data - tries MongoDB first, always saves audio files"""
        print(f"\n💾 Saving session {self.session_id}...")
        
        # Save audio files
        self.save_audio_files()
        
        # Try to save to MongoDB
        if self.transcript_log:
            mongodb_saved = self.save_session_to_mongodb()
            if not mongodb_saved:
                # Fallback: save transcript as JSON file
                transcript_file = self.session_dir / f"transcript_{self.session_id}.json"
                try:
                    import json
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "session_id": self.session_id,
                            "start_time": self.session_start.isoformat(),
                            "end_time": datetime.now(timezone.utc).isoformat(),
                            "transcript": [
                                {
                                    "timestamp": entry["timestamp"].isoformat(),
                                    "type": entry["type"],
                                    "content": entry["content"]
                                }
                                for entry in self.transcript_log
                            ]
                        }, indent=2, ensure_ascii=False)
                    print(f"💾 Transcript saved as JSON: {transcript_file}")
                except Exception as e:
                    print(f"❌ Error saving transcript JSON: {e}")
        
        print(f"✅ Session data saved in: {self.session_dir}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        
        # Initialize session logger
        self.session_logger = SessionLogger()
        self.input_audio_chunk_index = 0
        self.output_audio_chunk_index = 0

    async def voice_assistant_feedback(self):
        """Provides feedback that the voice assistant is listening"""
        print("🎤 Voice Assistant Ready - Start speaking! (Press Ctrl+C to exit)")
        print("📢 Make sure you're wearing headphones to prevent feedback")
        print("🔊 Listening for your voice...")
        
        # Keep the assistant running indefinitely - exit with Ctrl+C
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("\n👋 Voice assistant stopped.")
            raise

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    def find_best_microphone(self):
        """Find the best available microphone, preferring headphones"""
        print("🔍 Searching for microphones...")
        
        # Prioritize headphone microphones
        headphone_keywords = ['headphone', 'headset', 'bluetooth', 'bt', 'wireless', 'sony', 'bose', 'apple', 'airpods']
        
        devices = []
        for i in range(pya.get_device_count()):
            try:
                device_info = pya.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append((i, device_info))
                    device_name = device_info['name'].lower()
                    
                    # Check for headphone indicators
                    if any(keyword in device_name for keyword in headphone_keywords):
                        print(f"🎧 Found headphone microphone: {device_info['name']} (Device {i})")
                        return i, device_info
                        
            except Exception as e:
                continue
        
        # If no headphone found, try each available device
        print("⚠️  No headphone microphone detected. Available devices:")
        for i, device_info in devices:
            print(f"   Device {i}: {device_info['name']}")
        
        # Use default or first available
        try:
            default_input = pya.get_default_input_device_info()
            print(f"📡 Using default: {default_input['name']} (Device {default_input['index']})")
            return default_input['index'], default_input
        except:
            if devices:
                i, device_info = devices[0]
                print(f"📡 Using first available: {device_info['name']} (Device {i})")
                return i, device_info
            else:
                raise Exception("No microphone found!")

    async def listen_audio(self):
        try:
            device_index, mic_info = self.find_best_microphone()
        except Exception as e:
            print(f"❌ Microphone initialization failed: {e}")
            print("🎙️ Please ensure you have a microphone connected and configured.")
            print("💡 You can use 'test_microphones.py' to check your audio devices.")
            return  # Exit if no microphone is found

        # Test different channel configurations
        for channels in [1, 2]:
            try:
                print(f"🎤 Testing {channels} channel(s) on {mic_info['name']}")
                self.audio_stream = await asyncio.to_thread(
                    pya.open,
                    format=FORMAT,
                    channels=channels,
                    rate=SEND_SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK_SIZE,
                )
                print(f"✅ Successfully opened microphone with {channels} channel(s)")
                break  # Success
            except Exception as e:
                print(f"❌ Failed to open with {channels} channel(s): {e}")
                if channels == 2:  # Last attempt failed
                    print("\n---\n")
                    print("❌ Critical Error: Could not open microphone.")
                    print("This might be due to a few reasons:")
                    print("  1. Another application is using the microphone.")
                    print("  2. The microphone is not properly connected or configured.")
                    print("  3. Incorrect audio drivers.")
                    print("\n💡 Troubleshooting:")
                    print("  - Try running the 'test_microphones.py' script.")
                    print("  - Close other apps that might use the mic (Zoom, etc.).")
                    print("  - Check your system's audio settings.")
                    print("--- ---")
                    raise Exception("Failed to open audio stream after multiple attempts.")
        
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
            
        print("🔊 Monitoring audio levels... (speak now to test)")
        audio_level_check_count = 0
        
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            
            # Monitor audio levels for the first few seconds
            if audio_level_check_count < 50:  # Check for ~3 seconds
                import struct
                audio_data = struct.unpack(f'{len(data)//2}h', data)
                max_amplitude = max(abs(sample) for sample in audio_data) if audio_data else 0
                
                if audio_level_check_count % 10 == 0:  # Print every ~0.6 seconds
                    level_bars = "█" * min(20, max_amplitude // 1000)
                    print(f"🔊 Audio level: {level_bars} ({max_amplitude})")
                    
                if max_amplitude > 1000:  # Some audio detected
                    print("✅ Audio input detected! Voice assistant is working.")
                
                audio_level_check_count += 1
                
            # Save input audio chunk for session logging
            self.session_logger.save_audio_chunk(data, "input", self.input_audio_chunk_index)
            self.input_audio_chunk_index += 1
            
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            current_response_text = ""
            async for response in turn:
                if data := response.data:
                    # Save output audio chunk for session logging
                    self.session_logger.save_audio_chunk(data, "output", self.output_audio_chunk_index)
                    self.output_audio_chunk_index += 1
                    
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    current_response_text += text
                    print(text, end="")
            
            # Log the complete response text if we have any
            if current_response_text.strip():
                self.session_logger.log_transcript("assistant_response", current_response_text.strip())

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                voice_feedback_task = tg.create_task(self.voice_assistant_feedback())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await voice_feedback_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)
        finally:
            # Save session data before exiting
            try:
                self.session_logger.save_session()
                self.session_logger.close()
            except Exception as e:
                print(f"⚠️  Error saving session: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())