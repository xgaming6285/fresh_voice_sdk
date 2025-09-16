# -*- coding: utf-8 -*-
"""
Windows VoIP Voice Agent - Direct integration with Gate VoIP system
Works directly on Windows without requiring Linux or WSL
"""

import asyncio
import json
import logging
import uuid
import wave
import io
import base64
import socket
import struct
import threading
import time
import audioop
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import queue

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from pydub import AudioSegment
import pyaudio
import requests

# Import our existing voice agent components
from main import SessionLogger, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, FORMAT, CHANNELS
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    with open('asterisk_config.json', 'r') as f:
        return json.load(f)

config = load_config()

app = FastAPI(title="Windows VoIP Voice Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions: Dict[str, Dict] = {}
voice_client = genai.Client(http_options={"api_version": "v1beta"})

VOICE_CONFIG = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "Puck"
            }
        }
    },
    "system_instruction": {
        "parts": [
            {
                "text": "You are a helpful AI voice assistant answering phone calls. Start by greeting the caller with 'Hello, how can I help you today?' Keep responses conversational and concise since this is a voice-only interaction. Speak clearly with good enunciation, at a moderate pace, and avoid mumbling or speaking too softly. Project your voice as if speaking over a phone line."
            }
        ]
    }
}
MODEL = "models/gemini-2.0-flash-live-001"

class RTPServer:
    """RTP server to handle audio streams for voice calls"""
    
    def __init__(self, local_ip: str, rtp_port: int = 5004):
        self.local_ip = local_ip
        self.rtp_port = rtp_port
        self.socket = None
        self.running = False
        self.sessions = {}  # session_id -> RTPSession
        
    def start(self):
        """Start RTP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.local_ip, self.rtp_port))
            self.running = True
            
            logger.info(f"🎵 RTP server started on {self.local_ip}:{self.rtp_port}")
            
            # Start listening thread
            threading.Thread(target=self._listen_loop, daemon=True).start()
            
        except Exception as e:
            logger.error(f"❌ Failed to start RTP server: {e}")
    
    def _listen_loop(self):
        """Main RTP listening loop"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                
                # Parse RTP header (minimum 12 bytes)
                if len(data) < 12:
                    continue
                
                # RTP header format:
                # 0-1: V(2) P(1) X(1) CC(4) M(1) PT(7)
                # 2-3: Sequence number
                # 4-7: Timestamp  
                # 8-11: SSRC
                
                rtp_header = struct.unpack('!BBHII', data[:12])
                version = (rtp_header[0] >> 6) & 0x3
                padding = (rtp_header[0] >> 5) & 0x1
                extension = (rtp_header[0] >> 4) & 0x1
                cc = rtp_header[0] & 0xF
                marker = (rtp_header[1] >> 7) & 0x1
                payload_type = rtp_header[1] & 0x7F
                sequence = rtp_header[2]
                timestamp = rtp_header[3]
                ssrc = rtp_header[4]
                
                # Extract audio payload (skip header + CSRC if present)
                header_length = 12 + (cc * 4)
                if len(data) > header_length:
                    audio_payload = data[header_length:]
                    
                    # Find session for this SSRC/address
                    session_found = False
                    for session_id, rtp_session in self.sessions.items():
                        # Match by IP address only (port can be different for RTP vs SIP)
                        if rtp_session.remote_addr[0] == addr[0]:
                            logger.info(f"🎵 Received RTP audio from {addr}: {len(audio_payload)} bytes, PT={payload_type}")
                            
                            # Update the RTP session's actual RTP address if it changed
                            if rtp_session.remote_addr != addr:
                                logger.info(f"📍 Updating RTP address from {rtp_session.remote_addr} to {addr}")
                                rtp_session.remote_addr = addr
                            
                            rtp_session.process_incoming_audio(audio_payload, payload_type, timestamp)
                            session_found = True
                            break
                    
                    if not session_found:
                        # No session found for this address
                        logger.warning(f"⚠️ Received RTP audio from unknown address {addr}: {len(audio_payload)} bytes")
                
            except Exception as e:
                logger.error(f"Error in RTP listener: {e}")
    
    def create_session(self, session_id: str, remote_addr, voice_session):
        """Create RTP session for a call"""
        rtp_session = RTPSession(session_id, remote_addr, self.socket, voice_session)
        self.sessions[session_id] = rtp_session
        logger.info(f"🎵 Created RTP session {session_id} for {remote_addr}")
        return rtp_session
    
    def remove_session(self, session_id: str):
        """Remove RTP session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🎵 Removed RTP session {session_id}")
    
    def stop(self):
        """Stop RTP server"""
        self.running = False
        if self.socket:
            self.socket.close()

class RTPSession:
    """Individual RTP session for a call"""
    
    def __init__(self, session_id: str, remote_addr, rtp_socket, voice_session):
        self.session_id = session_id
        self.remote_addr = remote_addr
        self.rtp_socket = rtp_socket
        self.voice_session = voice_session
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = hash(session_id) & 0xFFFFFFFF
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.processing = False
        
        # Output audio queue for paced delivery
        self.output_queue = queue.Queue()
        self.output_thread = None
        
        # Adaptive jitter buffer
        self.jitter_buffer = []
        self.target_jitter_delay = 60  # Start with 60ms buffer
        self.min_jitter_delay = 40  # Minimum 40ms
        self.max_jitter_delay = 200  # Maximum 200ms
        self.jitter_stats = {"late": 0, "on_time": 0, "early": 0}
        
        # Audio level tracking for AGC
        self.audio_level_history = []
        self.target_audio_level = -20  # dBFS
        
        # Crossfade buffer for smooth transitions
        self.last_audio_tail = b""  # Last 10ms of previous chunk
        
    def process_incoming_audio(self, audio_data: bytes, payload_type: int, timestamp: int):
        """Process incoming RTP audio packet"""
        try:
            # Convert payload based on type
            if payload_type == 0:  # PCMU/μ-law
                pcm_data = self.ulaw_to_pcm(audio_data)
            elif payload_type == 8:  # PCMA/A-law  
                pcm_data = self.alaw_to_pcm(audio_data)
            else:
                # Assume it's already PCM
                pcm_data = audio_data
            
            # Queue audio for processing
            self.audio_queue.put(pcm_data)
            logger.info(f"🎤 Queued {len(pcm_data)} bytes of PCM audio for processing")
            
            # Start processing if not already running
            if not self.processing:
                self.processing = True
                threading.Thread(target=self._process_audio_queue, daemon=True).start()
                # Start output thread for paced audio delivery
                threading.Thread(target=self._process_output_queue, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
    
    def _process_audio_queue(self):
        """Process queued audio through voice session with improved async handling"""
        audio_buffer = b""
        chunk_size = 3200  # 200ms at 8kHz, 16-bit = 3200 bytes - larger for less processing overhead
        
        # Create dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async processing in the dedicated loop
            loop.run_until_complete(self._async_process_audio_queue(audio_buffer, chunk_size))
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            loop.close()
            self.processing = False
    
    async def _async_process_audio_queue(self, audio_buffer: bytes, chunk_size: int):
        """Async version of audio queue processing with proper streaming"""
        try:
            # Initialize voice session first
            if not await self.voice_session.initialize_voice_session():
                logger.error("Failed to initialize voice session")
                return
                
            # Start receive task for continuous response handling
            receive_task = asyncio.create_task(self._continuous_receive_responses())
            
            # Use consistent chunk size for predictable latency
            # 80ms chunks provide good balance between latency and quality
            chunk_size = 1280  # 80ms at 8kHz, 16-bit = 1280 bytes
            min_chunk_size = 640  # 40ms minimum to avoid too small chunks
            
            # Track timing for consistent chunk delivery
            last_send_time = time.time()
            target_interval = 0.08  # 80ms between chunks
            
            while self.processing:
                try:
                    # Get audio chunk with timeout - use asyncio-compatible approach
                    try:
                        # Use a small timeout to check if processing should continue
                        audio_chunk = self.audio_queue.get_nowait()
                        audio_buffer += audio_chunk
                    except queue.Empty:
                        # No audio available, wait a bit and continue
                        await asyncio.sleep(0.005)  # Even shorter sleep for lower latency
                        
                        # Process remaining buffer if we have some audio but not enough for full chunk
                        if len(audio_buffer) >= min_chunk_size and self.voice_session.gemini_session:
                            # Pad to chunk size for consistency
                            if len(audio_buffer) < chunk_size:
                                # Pad with silence instead of waiting
                                padding_needed = chunk_size - len(audio_buffer)
                                audio_buffer += b'\x00\x00' * (padding_needed // 2)
                            
                            await self._send_audio_to_gemini(audio_buffer[:chunk_size])
                            audio_buffer = audio_buffer[chunk_size:] if len(audio_buffer) > chunk_size else b""
                        continue
                    
                    # Process when we have enough audio
                    while len(audio_buffer) >= chunk_size:
                        # Ensure consistent timing between chunks
                        current_time = time.time()
                        time_since_last = current_time - last_send_time
                        if time_since_last < target_interval:
                            await asyncio.sleep(target_interval - time_since_last)
                        
                        chunk_to_process = audio_buffer[:chunk_size]
                        audio_buffer = audio_buffer[chunk_size:]
                        
                        # Send audio to Gemini (don't wait for response)
                        if self.voice_session.gemini_session:
                            await self._send_audio_to_gemini(chunk_to_process)
                            last_send_time = time.time()
                        
                except Exception as e:
                    logger.error(f"Error in async audio processing: {e}")
                    await asyncio.sleep(0.05)  # Reduced pause before retrying
                    
        except Exception as e:
            logger.error(f"Error in async audio queue processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            # Cancel receive task when done
            if 'receive_task' in locals():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
    
    async def _send_audio_to_gemini(self, audio_chunk: bytes):
        """Send audio chunk to Gemini without waiting for response"""
        try:
            if not self.voice_session.gemini_session:
                logger.warning("No Gemini session available")
                return
                
            # Convert telephony audio to Gemini format
            processed_audio = self.voice_session.convert_telephony_to_gemini(audio_chunk)
            logger.info(f"🎙️ Sending audio chunk to Gemini: {len(audio_chunk)} bytes → {len(processed_audio)} bytes")
            
            # Send audio to Gemini
            await self.voice_session.gemini_session.send(
                input={"data": processed_audio, "mime_type": "audio/pcm"}
            )
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")
            
    async def _continuous_receive_responses(self):
        """Continuously receive responses from Gemini in a separate task"""
        logger.info("🎧 Starting continuous response receiver")
        
        while self.processing and self.voice_session.gemini_session:
            try:
                # Get response from Gemini - this is a continuous stream
                turn = self.voice_session.gemini_session.receive()
                
                async for response in turn:
                    try:
                        # First check for server_content which contains the actual audio
                        if hasattr(response, 'server_content') and response.server_content:
                            if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        if hasattr(part.inline_data, 'mime_type') and 'audio' in part.inline_data.mime_type:
                                            audio_data = part.inline_data.data
                                            if isinstance(audio_data, str):
                                                # Base64 encoded audio
                                                try:
                                                    audio_bytes = base64.b64decode(audio_data)
                                                    logger.info(f"📥 Received {len(audio_bytes)} bytes of audio from Gemini")
                                                    # Convert and send in smaller chunks
                                                    telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_bytes)
                                                    
                                                    # Buffer small chunks to avoid fragmentation
                                                    if len(telephony_audio) < 160:  # Less than 20ms
                                                        # Store in buffer for later
                                                        if not hasattr(self, '_audio_buffer'):
                                                            self._audio_buffer = b""
                                                        self._audio_buffer += telephony_audio
                                                        
                                                        # Send when we have enough
                                                        if len(self._audio_buffer) >= 160:
                                                            self.send_audio(self._audio_buffer)
                                                            self._audio_buffer = b""
                                                    else:
                                                        # Send immediately if large enough
                                                        self.send_audio(telephony_audio)
                                                except Exception as e:
                                                    logger.error(f"Error decoding base64 audio: {e}")
                                            elif isinstance(audio_data, bytes):
                                                logger.info(f"📥 Received {len(audio_data)} bytes of audio from Gemini")
                                                # Convert and send in smaller chunks to avoid overwhelming the receiver
                                                telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_data)
                                                
                                                # Buffer small chunks to avoid fragmentation
                                                if len(telephony_audio) < 160:  # Less than 20ms
                                                    # Store in buffer for later
                                                    if not hasattr(self, '_audio_buffer'):
                                                        self._audio_buffer = b""
                                                    self._audio_buffer += telephony_audio
                                                    
                                                    # Send when we have enough
                                                    if len(self._audio_buffer) >= 160:
                                                        self.send_audio(self._audio_buffer)
                                                        self._audio_buffer = b""
                                                else:
                                                    # Send immediately if large enough
                                                    self.send_audio(telephony_audio)
                                    
                                    # Also handle text parts for logging
                                    if hasattr(part, 'text') and part.text:
                                        self.voice_session.session_logger.log_transcript(
                                            "assistant_response", part.text.strip()
                                        )
                                        logger.info(f"AI Response: {part.text.strip()}")
                        
                        # Also check direct data field (older format)
                        elif hasattr(response, 'data') and response.data:
                            if isinstance(response.data, bytes):
                                logger.info(f"📥 Received {len(response.data)} bytes of audio from Gemini (direct)")
                                telephony_audio = self.voice_session.convert_gemini_to_telephony(response.data)
                                # Apply same buffering logic
                                if len(telephony_audio) < 160:  # Less than 20ms
                                    if not hasattr(self, '_audio_buffer'):
                                        self._audio_buffer = b""
                                    self._audio_buffer += telephony_audio
                                    if len(self._audio_buffer) >= 160:
                                        self.send_audio(self._audio_buffer)
                                        self._audio_buffer = b""
                                else:
                                    self.send_audio(telephony_audio)
                            elif isinstance(response.data, str):
                                try:
                                    audio_bytes = base64.b64decode(response.data)
                                    logger.info(f"📥 Received {len(audio_bytes)} bytes of audio from Gemini (direct base64)")
                                    telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_bytes)
                                    # Apply same buffering logic
                                    if len(telephony_audio) < 160:  # Less than 20ms
                                        if not hasattr(self, '_audio_buffer'):
                                            self._audio_buffer = b""
                                        self._audio_buffer += telephony_audio
                                        if len(self._audio_buffer) >= 160:
                                            self.send_audio(self._audio_buffer)
                                            self._audio_buffer = b""
                                    else:
                                        self.send_audio(telephony_audio)
                                except:
                                    pass
                        
                        # Handle text response
                        if hasattr(response, 'text') and response.text:
                            self.voice_session.session_logger.log_transcript(
                                "assistant_response", response.text
                            )
                            logger.info(f"AI Response: {response.text}")
                            
                    except Exception as e:
                        logger.error(f"Error processing response item: {e}")
                        # Continue processing other responses
                        continue
                        
            except Exception as e:
                if self.processing:  # Only log if we're still processing
                    logger.error(f"Error receiving from Gemini: {e}")
                    await asyncio.sleep(0.05)  # Reduced pause before retrying
                    
        logger.info("🎧 Stopped continuous response receiver")
    
    # Removed _process_chunk - we use continuous streaming instead
    
    def send_audio(self, audio_data: bytes):
        """Queue audio for paced RTP transmission with proper chunking"""
        try:
            # Convert PCM to μ-law for transmission
            ulaw_data = self.pcm_to_ulaw(audio_data)
            
            # Split large audio chunks into smaller pieces for smoother delivery
            # This prevents large chunks from causing audio glitches
            chunk_size = 320  # 40ms of μ-law audio at 8kHz
            
            if len(ulaw_data) > chunk_size:
                # Split into smaller chunks
                for i in range(0, len(ulaw_data), chunk_size):
                    chunk = ulaw_data[i:i + chunk_size]
                    self.output_queue.put(chunk)
                logger.info(f"📤 Queued {len(ulaw_data)} bytes as {len(ulaw_data)//chunk_size + 1} chunks for RTP transmission")
            else:
                # Small enough to send as-is
                self.output_queue.put(ulaw_data)
                logger.info(f"📤 Queued {len(ulaw_data)} bytes for RTP transmission")
            
        except Exception as e:
            logger.error(f"Error queueing audio: {e}")
    
    def _process_output_queue(self):
        """Process output queue with adaptive RTP pacing and intelligent buffering"""
        packet_size = 160  # 20ms of μ-law audio at 8kHz
        packet_duration_s = 0.020  # 20ms per packet
        
        # Use high-precision timing
        import time
        if hasattr(time, 'perf_counter'):
            get_time = time.perf_counter  # More precise on Windows
        else:
            get_time = time.time
        
        last_packet_time = get_time()
        
        # Buffer to accumulate audio for smoother delivery
        audio_buffer = b""
        buffer_threshold = packet_size * 3  # Buffer 60ms for better smoothing
        
        # Silence/comfort noise for gaps
        silence_counter = 0
        max_silence_packets = 5  # Send up to 100ms of silence
        
        # Adaptive pacing variables
        jitter_buffer_size = 2  # Start with 2 packets buffer
        max_jitter_buffer = 5   # Maximum 5 packets
        
        # Statistics for adaptive adjustment
        late_packets = 0
        total_packets = 0
        
        while self.processing:
            try:
                # Adaptive timeout based on buffer state
                buffer_packets = len(audio_buffer) // packet_size
                timeout = 0.005 if buffer_packets < jitter_buffer_size else 0.020
                
                # Get audio from queue with adaptive timeout
                try:
                    audio_data = self.output_queue.get(timeout=timeout)
                    audio_buffer += audio_data
                    silence_counter = 0  # Reset silence counter when we get audio
                    
                    # Adapt jitter buffer size based on buffer state
                    if buffer_packets > max_jitter_buffer and jitter_buffer_size > 2:
                        jitter_buffer_size -= 1  # Reduce buffer if too full
                    
                except queue.Empty:
                    # If we have buffered data and no new data coming, send what we have
                    if len(audio_buffer) >= packet_size:
                        # Check if we're running low on buffer
                        if buffer_packets < 2 and jitter_buffer_size < max_jitter_buffer:
                            jitter_buffer_size += 1  # Increase buffer if running low
                    else:
                        # Generate comfort noise to maintain stream continuity
                        if silence_counter < max_silence_packets:
                            comfort_noise = self._generate_comfort_noise(packet_size)
                            audio_buffer += comfort_noise
                            silence_counter += 1
                        else:
                            # During silence, use longer sleep to save CPU
                            time.sleep(0.010)
                            continue
                
                # Send buffered audio when we have enough
                while len(audio_buffer) >= packet_size:
                    chunk = audio_buffer[:packet_size]
                    audio_buffer = audio_buffer[packet_size:]
                    
                    # Create RTP packet
                    rtp_packet = self._create_rtp_packet(chunk, payload_type=0)
                    
                    # High-precision timing with drift compensation
                    current_time = get_time()
                    time_since_last = current_time - last_packet_time
                    
                    # Track if we're late
                    if time_since_last > packet_duration_s * 1.1:  # 10% tolerance
                        late_packets += 1
                    
                    total_packets += 1
                    
                    # Adjust timing to prevent drift
                    target_time = last_packet_time + packet_duration_s
                    sleep_time = target_time - current_time
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        last_packet_time = target_time
                    else:
                        # We're behind schedule, catch up
                        last_packet_time = current_time
                    
                    # Send packet
                    self.rtp_socket.sendto(rtp_packet, self.remote_addr)
                    
                    # Log statistics periodically
                    if total_packets % 100 == 0 and total_packets > 0:
                        late_ratio = late_packets / total_packets
                        logger.debug(f"RTP stats: {total_packets} packets, {late_ratio:.1%} late, buffer: {jitter_buffer_size}")
                        # Reset counters
                        late_packets = 0
                        total_packets = 0
                    
            except Exception as e:
                logger.error(f"Error in output queue processing: {e}")
    
    def _create_rtp_packet(self, payload: bytes, payload_type: int = 0):
        """Create RTP packet with payload"""
        # RTP header
        version = 2
        padding = 0
        extension = 0
        cc = 0
        marker = 0
        
        # Pack header
        byte0 = (version << 6) | (padding << 5) | (extension << 4) | cc
        byte1 = (marker << 7) | payload_type
        
        header = struct.pack('!BBHII', 
                           byte0, byte1, 
                           self.sequence_number, 
                           self.timestamp, 
                           self.ssrc)
        
        # Increment sequence and timestamp
        self.sequence_number = (self.sequence_number + 1) & 0xFFFF
        # For μ-law (G.711), timestamp increments by number of samples
        # At 8kHz, each byte represents one sample
        # This ensures proper timestamp progression for audio synchronization
        self.timestamp = (self.timestamp + len(payload)) & 0xFFFFFFFF
        
        return header + payload
    
    def ulaw_to_pcm(self, ulaw_data: bytes) -> bytes:
        """Convert μ-law to 16-bit PCM"""
        return audioop.ulaw2lin(ulaw_data, 2)
    
    def alaw_to_pcm(self, alaw_data: bytes) -> bytes:
        """Convert A-law to 16-bit PCM"""
        return audioop.alaw2lin(alaw_data, 2)
    
    def pcm_to_ulaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to μ-law with proper handling"""
        try:
            # Ensure the PCM data is properly formatted before conversion
            # This helps prevent conversion artifacts
            # μ-law expects 16-bit signed PCM input
            
            # Apply a slight attenuation to prevent clipping during conversion
            # μ-law compression can amplify near-peak signals
            import struct
            import array
            
            # Convert bytes to array of 16-bit samples
            samples = array.array('h', pcm_data)
            
            # Apply soft limiting to prevent μ-law conversion artifacts
            max_val = 32767 * 0.95  # Leave 5% headroom
            for i in range(len(samples)):
                if samples[i] > max_val:
                    samples[i] = int(max_val)
                elif samples[i] < -max_val:
                    samples[i] = int(-max_val)
            
            # Convert back to bytes
            pcm_data = samples.tobytes()
            
            # Convert to μ-law
            return audioop.lin2ulaw(pcm_data, 2)
        except Exception as e:
            logger.error(f"Error in PCM to μ-law conversion: {e}")
            # Fallback to simple conversion
            return audioop.lin2ulaw(pcm_data, 2)
    
    def _generate_comfort_noise(self, length: int) -> bytes:
        """Generate pink noise for comfort noise (μ-law encoded)"""
        import random
        import struct
        
        # Pink noise generation using Voss-McCartney algorithm
        # This creates more natural sounding background noise
        num_sources = 16
        sources = [0] * num_sources
        
        samples = []
        for _ in range(length):
            # Update random sources
            for i in range(num_sources):
                if random.random() < 1.0 / (1 << i):
                    sources[i] = random.uniform(-1, 1)
            
            # Sum all sources and scale to very low level
            pink_sample = sum(sources) / num_sources * 0.002  # Very quiet
            
            # Convert to 16-bit PCM
            pcm_sample = int(pink_sample * 32767)
            pcm_sample = max(-32768, min(32767, pcm_sample))
            
            # Convert to μ-law
            pcm_bytes = struct.pack('h', pcm_sample)
            ulaw_byte = audioop.lin2ulaw(pcm_bytes, 2)
            samples.append(ulaw_byte[0])
        
        return bytes(samples)

class WindowsSIPHandler:
    """Simple SIP handler for Windows - interfaces with Gate VoIP system"""
    
    def __init__(self):
        self.config = config
        self.local_ip = self.config['local_ip']
        self.gate_ip = self.config['host']
        self.sip_port = self.config['sip_port']
        self.phone_number = self.config['phone_number']
        self.running = False
        self.socket = None
        self.registration_attempts = 0
        self.max_registration_attempts = 3
        self.last_nonce = None
        
        # Initialize RTP server
        self.rtp_server = RTPServer(self.local_ip)
        
    def start_sip_listener(self):
        """Start listening for SIP messages from Gate VoIP"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.local_ip, self.sip_port))
            self.running = True
            
            logger.info(f"SIP listener started on {self.local_ip}:{self.sip_port}")
            logger.info(f"Listening for calls from Gate VoIP at {self.gate_ip}")
            
            # Start RTP server
            self.rtp_server.start()
            
            # Start listening thread
            threading.Thread(target=self._listen_loop, daemon=True).start()
            
            # Note: We don't need to register with Gate VoIP since it registers with us as a trunk
            logger.info("🎯 Acting as SIP trunk - waiting for Gate VoIP to register with us")
            
        except Exception as e:
            logger.error(f"Failed to start SIP listener: {e}")
    
    def _listen_loop(self):
        """Main SIP listening loop"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = data.decode('utf-8')
                
                # Determine message type for better logging
                message_type = "UNKNOWN"
                first_line = message.split('\n')[0].strip() if message.split('\n') else ""
                
                if first_line.startswith('INVITE'):
                    message_type = "INVITE (Incoming Call)"
                elif first_line.startswith('OPTIONS'):
                    message_type = "OPTIONS (Keep-alive)"
                elif first_line.startswith('BYE'):
                    message_type = "BYE (Call End)"
                elif first_line.startswith('ACK'):
                    message_type = "ACK (Acknowledgment)"
                elif first_line.startswith('INFO'):
                    message_type = "INFO (Call Update)"
                elif first_line.startswith('REGISTER'):
                    message_type = "REGISTER (Registration)"
                elif first_line.startswith('SIP/2.0 200'):
                    message_type = "200 OK (Success Response)"
                elif first_line.startswith('SIP/2.0 401'):
                    message_type = "401 Unauthorized (Auth Challenge)"
                elif first_line.startswith('SIP/2.0 403'):
                    message_type = "403 Forbidden (Auth Failed)"
                elif first_line.startswith('SIP/2.0'):
                    message_type = f"SIP Response ({first_line.split(' ', 2)[1] if len(first_line.split(' ', 2)) > 1 else 'Unknown'})"
                
                logger.info(f"📞 Received SIP {message_type} from {addr}")
                logger.debug(f"Full message: {message[:200] + '...' if len(message) > 200 else message}")
                
                # Handle different SIP message types (check first line for method)
                first_line = message.split('\n')[0].strip()
                if first_line.startswith('INVITE'):
                    self._handle_invite(message, addr)
                elif first_line.startswith('OPTIONS'):
                    self._handle_options(message, addr)
                elif first_line.startswith('INFO'):
                    self._handle_info(message, addr)
                elif first_line.startswith('BYE'):
                    self._handle_bye(message, addr)
                elif first_line.startswith('ACK'):
                    self._handle_ack(message, addr)
                elif first_line.startswith('REGISTER'):
                    self._handle_register(message, addr)
                elif first_line.startswith('SIP/2.0') and ('200 OK' in first_line or '401 Unauthorized' in first_line or '403 Forbidden' in first_line):
                    self._handle_sip_response(message, addr)
                else:
                    logger.warning(f"⚠️  Unhandled SIP message type: {first_line}")
                    
            except Exception as e:
                logger.error(f"Error in SIP listener: {e}")
    
    def _handle_invite(self, message: str, addr):
        """Handle incoming INVITE (new call)"""
        try:
            logger.info("📞 ================== INCOMING CALL ==================")
            logger.debug(f"INVITE message from {addr}:\n{message}")
            
            # Parse caller ID from SIP message
            caller_id = "Unknown"
            call_id = ""
            from_tag = ""
            
            for line in message.split('\n'):
                line = line.strip()
                if line.startswith('From:'):
                    # Extract caller ID and from tag
                    if '<sip:' in line:
                        start = line.find('<sip:') + 5
                        end = line.find('@', start)
                        if end > start:
                            caller_id = line[start:end]
                    if 'tag=' in line:
                        tag_start = line.find('tag=') + 4
                        tag_end = line.find(';', tag_start)
                        if tag_end == -1:
                            tag_end = len(line)
                        from_tag = line[tag_start:tag_end]
                elif line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
            
            logger.info(f"📞 Incoming call from {caller_id} (Call-ID: {call_id})")
            logger.info(f"📞 From address: {addr}")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Send 180 Ringing first
            logger.info("📤 Sending 180 Ringing...")
            ringing_response = self._create_sip_ringing_response(message, session_id)
            self.socket.sendto(ringing_response.encode(), addr)
            logger.info("🔔 180 Ringing sent")
            
            # Brief delay before 200 OK
            import time
            time.sleep(1)
            
            # Send 200 OK response
            logger.info("📤 Sending 200 OK response...")
            ok_response = self._create_sip_ok_response(message, session_id)
            logger.debug(f"200 OK response:\n{ok_response}")
            self.socket.sendto(ok_response.encode(), addr)
            logger.info("✅ 200 OK response sent")
            
            # Create voice session
            logger.info(f"🎙️  Creating voice session {session_id}")
            voice_session = WindowsVoiceSession(session_id, caller_id, self.phone_number)
            
            # Create RTP session for audio
            rtp_session = self.rtp_server.create_session(session_id, addr, voice_session)
            
            active_sessions[session_id] = {
                "voice_session": voice_session,
                "rtp_session": rtp_session,
                "caller_addr": addr,
                "status": "connecting",
                "call_start": datetime.now(timezone.utc),
                "call_id": call_id
            }
            
            # Start voice session in separate thread
            def start_voice_session():
                try:
                    asyncio.run(voice_session.initialize_voice_session())
                    # Mark as active once voice session is ready
                    if session_id in active_sessions:
                        active_sessions[session_id]["status"] = "active"
                        logger.info(f"🎯 Voice session {session_id} is now active and ready")
                except Exception as e:
                    logger.error(f"❌ Failed to start voice session: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Remove failed session
                    if session_id in active_sessions:
                        del active_sessions[session_id]
            
            threading.Thread(target=start_voice_session, daemon=True).start()
            logger.info("📞 ================================================")
            
        except Exception as e:
            logger.error(f"❌ Error handling INVITE: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _handle_info(self, message: str, addr):
        """Handle INFO messages (call updates)"""
        try:
            # Send 200 OK response for INFO messages
            ok_response = "SIP/2.0 200 OK\r\n\r\n"
            self.socket.sendto(ok_response.encode(), addr)
            logger.debug("Responded to INFO message")
        except Exception as e:
            logger.error(f"Error handling INFO: {e}")
    
    def _handle_options(self, message: str, addr):
        """Handle OPTIONS messages (keep-alive/capabilities)"""
        try:
            # Parse headers from the original OPTIONS request
            via_header = ""
            from_header = ""
            to_header = ""
            call_id = "options-keepalive"
            cseq_header = ""
            
            for line in message.split('\n'):
                line = line.strip()
                if line.startswith('Via:'):
                    via_header = line
                elif line.startswith('From:'):
                    from_header = line
                elif line.startswith('To:'):
                    to_header = line
                elif line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
                elif line.startswith('CSeq:'):
                    cseq_header = line
            
            # Send proper 200 OK response for OPTIONS messages
            ok_response = f"""SIP/2.0 200 OK
{via_header}
{from_header}
{to_header}
{cseq_header}
Call-ID: {call_id}
Contact: <sip:voice-agent@{self.local_ip}:{self.sip_port}>
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO
Accept: application/sdp
Accept-Language: en
Supported: replaces, timer
Server: WindowsVoiceAgent/1.0
Content-Length: 0

"""
            self.socket.sendto(ok_response.encode(), addr)
            logger.info(f"✅ Responded to OPTIONS keep-alive from {addr}")
        except Exception as e:
            logger.error(f"Error handling OPTIONS: {e}")
    
    def _handle_bye(self, message: str, addr):
        """Handle call termination"""
        try:
            # Find session for this address
            for session_id, session_data in list(active_sessions.items()):
                if session_data.get("caller_addr") == addr:
                    logger.info(f"Call ended: {session_id}")
                    
                    # Cleanup voice session
                    voice_session = session_data["voice_session"]
                    rtp_session = session_data.get("rtp_session")
                    
                    # Stop RTP processing first
                    if rtp_session:
                        rtp_session.processing = False
                    
                    try:
                        if hasattr(voice_session, 'cleanup'):
                            # Handle cleanup properly without creating async task in wrong context
                            voice_session.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up voice session: {e}")
                    
                    # Cleanup RTP session
                    self.rtp_server.remove_session(session_id)
                    
                    # Remove from active sessions
                    del active_sessions[session_id]
                    
                    # Send 200 OK
                    ok_response = "SIP/2.0 200 OK\r\n\r\n"
                    self.socket.sendto(ok_response.encode(), addr)
                    break
                    
        except Exception as e:
            logger.error(f"Error handling BYE: {e}")
    
    def _handle_ack(self, message: str, addr):
        """Handle ACK message"""
        logger.debug("Received ACK")
    
    def _handle_register(self, message: str, addr):
        """Handle incoming REGISTER from Gate VoIP (trunk registration)"""
        try:
            logger.info(f"🔗 Gate VoIP attempting to register as trunk from {addr}")
            
            # Parse headers from the REGISTER request
            via_header = ""
            from_header = ""
            to_header = ""
            call_id = "trunk-register"
            cseq_header = ""
            contact_header = ""
            
            for line in message.split('\n'):
                line = line.strip()
                if line.startswith('Via:'):
                    via_header = line
                elif line.startswith('From:'):
                    from_header = line
                elif line.startswith('To:'):
                    to_header = line
                elif line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
                elif line.startswith('CSeq:'):
                    cseq_header = line
                elif line.startswith('Contact:'):
                    contact_header = line
            
            # Add tag to To header if not present
            if 'tag=' not in to_header:
                to_header += f';tag={call_id[:8]}'
            
            # Send 200 OK response to accept the registration
            ok_response = f"""SIP/2.0 200 OK
{via_header}
{from_header}
{to_header}
{cseq_header}
Call-ID: {call_id}
{contact_header}
Server: WindowsVoiceAgent/1.0
Expires: 3600
Date: {time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())}
Content-Length: 0

"""
            
            self.socket.sendto(ok_response.encode(), addr)
            logger.info("✅ Accepted Gate VoIP trunk registration")
            logger.info("🎯 Voice agent is now registered as SIP trunk!")
            
        except Exception as e:
            logger.error(f"Error handling incoming REGISTER: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _create_sip_ringing_response(self, invite_message: str, session_id: str) -> str:
        """Create SIP 180 Ringing response"""
        # Parse the original INVITE to extract necessary headers
        call_id = session_id
        from_header = ""
        to_header = ""
        via_header = ""
        cseq_header = ""
        
        for line in invite_message.split('\n'):
            line = line.strip()
            if line.startswith('Call-ID:'):
                call_id = line.split(':', 1)[1].strip()
            elif line.startswith('From:'):
                from_header = line
            elif line.startswith('To:'):
                to_header = line + f';tag={session_id[:8]}'  # Add tag for To header
            elif line.startswith('Via:'):
                via_header = line
            elif line.startswith('CSeq:'):
                cseq_header = line
        
        # Create SIP 180 Ringing response
        response = f"""SIP/2.0 180 Ringing
{via_header}
{from_header}
{to_header}
{cseq_header}
Call-ID: {call_id}
Contact: <sip:voice-agent@{self.local_ip}:{self.sip_port}>
Server: WindowsVoiceAgent/1.0
Content-Length: 0

"""
        return response
    
    def _create_sip_ok_response(self, invite_message: str, session_id: str) -> str:
        """Create SIP 200 OK response"""
        # Parse the original INVITE to extract necessary headers
        call_id = session_id
        from_header = ""
        to_header = ""
        via_header = ""
        cseq_header = ""
        
        for line in invite_message.split('\n'):
            line = line.strip()
            if line.startswith('Call-ID:'):
                call_id = line.split(':', 1)[1].strip()
            elif line.startswith('From:'):
                from_header = line
            elif line.startswith('To:'):
                to_header = line + f';tag={session_id[:8]}'  # Add tag for To header
            elif line.startswith('Via:'):
                via_header = line
            elif line.startswith('CSeq:'):
                cseq_header = line
        
        # Create proper SIP 200 OK response with better SDP
        sdp_content = f"""v=0
o=voice-agent {int(time.time())} {int(time.time())} IN IP4 {self.local_ip}
s=Voice Agent Session
c=IN IP4 {self.local_ip}
t=0 0
m=audio 5004 RTP/AVP 0 8 18 101
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:18 G729/8000
a=rtpmap:101 telephone-event/8000
a=sendrecv
a=ptime:20"""

        response = f"""SIP/2.0 200 OK
{via_header}
{from_header}
{to_header}
{cseq_header}
Call-ID: {call_id}
Contact: <sip:voice-agent@{self.local_ip}:{self.sip_port}>
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO
Supported: replaces, timer
Server: WindowsVoiceAgent/1.0
Content-Type: application/sdp
Content-Length: {len(sdp_content)}

{sdp_content}
"""
        return response
    
    def make_outbound_call(self, phone_number: str) -> Optional[str]:
        """Initiate outbound call through Gate VoIP"""
        try:
            # Create INVITE message for outbound call
            session_id = str(uuid.uuid4())
            invite_message = f"""INVITE sip:{phone_number}@{self.gate_ip} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port}
From: <sip:voice-agent@{self.local_ip}>;tag={session_id[:8]}
To: <sip:{phone_number}@{self.gate_ip}>
Call-ID: {session_id}
CSeq: 1 INVITE
Contact: <sip:voice-agent@{self.local_ip}:{self.sip_port}>
User-Agent: WindowsVoiceAgent/1.0
Content-Type: application/sdp
Content-Length: 147

v=0
o=voice-agent 0 0 IN IP4 {self.local_ip}
s=Voice Agent Session
c=IN IP4 {self.local_ip}
t=0 0
m=audio 5004 RTP/AVP 0 8
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000

"""
            
            # Send INVITE
            self.socket.sendto(invite_message.encode(), (self.gate_ip, self.sip_port))
            
            logger.info(f"Initiated outbound call to {phone_number}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error making outbound call: {e}")
            return None
    
    def _register_with_gate(self):
        """Register voice agent with Gate VoIP system"""
        try:
            # Reset registration attempts
            self.registration_attempts = 0
            self.last_nonce = None
            
            logger.info("🔗 Attempting to register with Gate VoIP...")
            logger.info(f"   Username: {self.config.get('username', 'voice-agent')}")
            logger.info(f"   Gate IP: {self.gate_ip}")
            logger.info(f"   Local IP: {self.local_ip}")
            
            # First try without authentication
            call_id = str(uuid.uuid4())
            username = self.config.get('username', 'voice-agent')
            register_message = f"""REGISTER sip:{self.gate_ip}:{self.sip_port} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{call_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={call_id[:8]}
To: <sip:{username}@{self.gate_ip}>
Call-ID: {call_id}
CSeq: 1 REGISTER
Contact: <sip:{username}@{self.local_ip}:{self.sip_port};expires=3600>
User-Agent: WindowsVoiceAgent/1.0
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO
Supported: replaces, timer
Expires: 3600
Content-Length: 0

"""
            
            # Send REGISTER
            self.socket.sendto(register_message.encode(), (self.gate_ip, self.sip_port))
            logger.info(f"📡 Sent initial REGISTER request to {self.gate_ip}:{self.sip_port}")
            logger.debug(f"REGISTER message:\n{register_message}")
            
            # Wait for authentication challenge, then re-register
            # The actual authentication will be handled in _handle_sip_response
                
        except Exception as e:
            logger.error(f"❌ Failed to register with Gate: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _create_authenticated_register(self, challenge_response: str):
        """Create REGISTER with authentication after receiving 401"""
        try:
            # Parse authentication challenge
            import re
            realm_match = re.search(r'realm="([^"]*)"', challenge_response)
            nonce_match = re.search(r'nonce="([^"]*)"', challenge_response)
            algorithm_match = re.search(r'algorithm=([^,\s]+)', challenge_response)
            qop_match = re.search(r'qop="([^"]*)"', challenge_response)
            opaque_match = re.search(r'opaque="([^"]*)"', challenge_response)
            
            realm = realm_match.group(1) if realm_match else self.gate_ip
            nonce = nonce_match.group(1) if nonce_match else ""
            algorithm = algorithm_match.group(1) if algorithm_match else "MD5"
            qop = qop_match.group(1) if qop_match else None
            opaque = opaque_match.group(1) if opaque_match else None
            
            username = self.config.get('username', 'voice-agent')
            password = self.config.get('password', '')
            
            logger.info(f"🔐 Creating authenticated REGISTER")
            logger.info(f"   Username: {username}")
            logger.info(f"   Realm: {realm}")
            logger.info(f"   Algorithm: {algorithm}")
            logger.info(f"   QoP: {qop}")
            logger.info(f"   Nonce: {nonce[:20]}..." if nonce else "   Nonce: (empty)")
            
            # Create digest response with proper format
            import hashlib
            uri = f"sip:{self.gate_ip}"  # Don't include port for digest calculation
            method = "REGISTER"
            
            # Calculate HA1
            ha1_input = f"{username}:{realm}:{password}"
            ha1 = hashlib.md5(ha1_input.encode()).hexdigest()
            
            # Calculate HA2  
            ha2_input = f"{method}:{uri}"
            ha2 = hashlib.md5(ha2_input.encode()).hexdigest()
            
            logger.debug(f"🔐 Debug - HA1 input: {ha1_input}")
            logger.debug(f"🔐 Debug - HA2 input: {ha2_input}")
            logger.debug(f"🔐 Debug - HA1: {ha1}")
            logger.debug(f"🔐 Debug - HA2: {ha2}")
            
            # Calculate response
            if qop:
                # If qop is specified, use more complex calculation
                nc = "00000001"  # Nonce count
                cnonce = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
                response_input = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
                response_hash = hashlib.md5(response_input.encode()).hexdigest()
                logger.debug(f"🔐 Debug - Response input (with qop): {response_input}")
            else:
                # Simple digest
                response_input = f"{ha1}:{nonce}:{ha2}"
                response_hash = hashlib.md5(response_input.encode()).hexdigest()
                logger.debug(f"🔐 Debug - Response input: {response_input}")
            
            logger.debug(f"🔐 Debug - Final response hash: {response_hash}")
            
            # Generate new call-id for authenticated request
            call_id = str(uuid.uuid4())
            cseq_number = 2
            
            # Build authorization header
            auth_header = f'Digest username="{username}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response_hash}"'
            if algorithm:
                auth_header += f', algorithm={algorithm}'
            if opaque:
                auth_header += f', opaque="{opaque}"'
            if qop:
                nc = "00000001"
                cnonce = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
                auth_header += f', qop={qop}, nc={nc}, cnonce="{cnonce}"'
            
            register_message = f"""REGISTER sip:{self.gate_ip}:{self.sip_port} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{call_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={call_id[:8]}
To: <sip:{username}@{self.gate_ip}>
Call-ID: {call_id}
CSeq: {cseq_number} REGISTER
Contact: <sip:{username}@{self.local_ip}:{self.sip_port};expires=3600>
User-Agent: WindowsVoiceAgent/1.0
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO
Supported: replaces, timer
Authorization: {auth_header}
Expires: 3600
Content-Length: 0

"""
            
            # Send authenticated REGISTER
            self.socket.sendto(register_message.encode(), (self.gate_ip, self.sip_port))
            logger.info(f"🔑 Sent authenticated REGISTER request")
            logger.debug(f"Authorization header: {auth_header}")
            logger.debug(f"Full REGISTER message:\n{register_message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create authenticated REGISTER: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _handle_sip_response(self, message: str, addr):
        """Handle SIP responses (200 OK, 401 Unauthorized, etc.)"""
        try:
            first_line = message.split('\n')[0].strip()
            
            if '200 OK' in first_line:
                if 'REGISTER' in message:
                    logger.info("✅ Successfully registered with Gate VoIP!")
                    logger.info("🎯 Voice agent is now ready to receive calls")
                else:
                    logger.debug(f"Received 200 OK response: {first_line}")
            
            elif '401 Unauthorized' in first_line and 'REGISTER' in message:
                # Check if we've already tried too many times
                if self.registration_attempts >= self.max_registration_attempts:
                    logger.error("❌ Too many authentication attempts, stopping registration")
                    logger.error("💡 Check username/password in asterisk_config.json")
                    logger.error("💡 Run: python check_gate_password.py to verify/update password")
                    logger.error("💡 Check Gate VoIP web interface: http://192.168.50.50 > PBX Settings > Internal Phones > Extension 200")
                    return
                
                # Parse nonce to avoid duplicate authentication attempts
                import re
                nonce_match = re.search(r'nonce="([^"]*)"', message)
                current_nonce = nonce_match.group(1) if nonce_match else None
                
                if current_nonce == self.last_nonce:
                    logger.error("❌ Same authentication challenge received - password is incorrect")
                    logger.error("💡 Run: python check_gate_password.py to verify/update password")
                    logger.error("💡 Check Gate VoIP web interface: http://192.168.50.50 > PBX Settings > Internal Phones > Extension 200")
                    return
                
                self.last_nonce = current_nonce
                self.registration_attempts += 1
                logger.info(f"🔐 Received authentication challenge (attempt {self.registration_attempts}/{self.max_registration_attempts})")
                logger.info(f"🔐 Nonce: {current_nonce[:20]}..." if current_nonce else "🔐 No nonce found")
                
                # Send authenticated REGISTER
                self._create_authenticated_register(message)
                
            elif '403 Forbidden' in first_line:
                logger.error(f"❌ Authentication failed: {first_line}")
                logger.error("💡 Check username/password in asterisk_config.json")
                logger.error("💡 Verify extension 200 is properly configured in Gate VoIP")
                
            elif '404 Not Found' in first_line:
                logger.error(f"❌ User not found: {first_line}")
                logger.error("💡 Check if extension 200 exists in Gate VoIP configuration")
                
            else:
                logger.info(f"📨 SIP Response: {first_line}")
                if '401' in first_line or '403' in first_line or '404' in first_line:
                    logger.debug(f"Full response message:\n{message}")
                
        except Exception as e:
            logger.error(f"Error handling SIP response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def stop(self):
        """Stop SIP handler"""
        self.running = False
        if self.socket:
            self.socket.close()
        
        # Stop RTP server
        self.rtp_server.stop()

class WindowsVoiceSession:
    """Voice session for Windows - simpler than Linux version"""
    
    def __init__(self, session_id: str, caller_id: str, called_number: str):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        self.session_logger = SessionLogger()
        self.voice_session = None
        self.gemini_session = None  # The actual session object from the context manager
        
        # Connection management
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.connection_backoff = 1.0  # seconds
        self.last_connection_attempt = 0
        
        # Log call start
        self.session_logger.log_transcript(
            "system", 
            f"Call started - From: {caller_id}, To: {called_number}"
        )
    
    async def initialize_voice_session(self):
        """Initialize the Google Gemini voice session with improved error handling"""
        current_time = time.time()
        
        # Check if we should wait before retrying
        if (current_time - self.last_connection_attempt) < self.connection_backoff:
            logger.info(f"Waiting {self.connection_backoff}s before retry...")
            await asyncio.sleep(self.connection_backoff)
        
        # Check if we've exceeded max attempts
        if self.connection_attempts >= self.max_connection_attempts:
            logger.error(f"❌ Max connection attempts ({self.max_connection_attempts}) exceeded for session {self.session_id}")
            return False
        
        self.connection_attempts += 1
        self.last_connection_attempt = time.time()
        
        try:
            logger.info(f"Attempting to connect to Gemini live session (attempt {self.connection_attempts}/{self.max_connection_attempts})...")
            logger.info(f"Using model: {MODEL}")
            logger.info(f"Voice config: {VOICE_CONFIG}")
            
            # Create the connection with timeout
            try:
                # Create the context manager
                context_manager = voice_client.aio.live.connect(
                    model=MODEL, 
                    config=VOICE_CONFIG
                )
                
                # Enter the context manager to get the actual session
                self.gemini_session = await context_manager.__aenter__()
                self.voice_session = context_manager  # Store context manager for cleanup
                
                logger.info(f"✅ Voice session connected successfully for {self.session_id}")
                logger.info(f"📞 Call from {self.caller_id} is now ready for voice interaction")
                
            except asyncio.TimeoutError:
                logger.error("❌ Connection to Gemini timed out")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to create Gemini connection: {e}")
                return False
            
            # Reset connection attempts on success
            self.connection_attempts = 0
            self.connection_backoff = 1.0
            
            # Wait a bit to ensure the connection is fully established
            await asyncio.sleep(0.1)
            
            # Skip the initial greeting trigger to reduce startup delay
            # The AI will respond naturally when it receives actual caller audio
            logger.info(f"🎤 Voice session ready - waiting for caller audio...")
            
            logger.info(f"🎤 Waiting for caller audio to trigger AI responses...")
            
            # Log successful initialization
            self.session_logger.log_transcript(
                "system", 
                f"Voice session initialized - ready for conversation"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize voice session (attempt {self.connection_attempts}): {e}")
            
            # Log specific error types for better debugging
            error_type = type(e).__name__
            if "websocket" in str(e).lower():
                logger.error("🔌 WebSocket connection error - check network connectivity")
            elif "auth" in str(e).lower() or "permission" in str(e).lower():
                logger.error("🔑 Authentication error - check API key and permissions")
            elif "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.error("⚠️ Rate limiting or quota exceeded")
            else:
                logger.error(f"🔧 Generic error ({error_type}): {str(e)}")
            
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.voice_session = None
            self.gemini_session = None
            
            # Exponential backoff for retries
            self.connection_backoff = min(self.connection_backoff * 2, 30.0)  # Max 30 seconds
            
            return False
    
    async def _receive_response(self) -> bytes:
        """Helper method to receive response from Gemini with proper error handling"""
        response_audio = b""
        
        try:
            # Get a turn from the session
            turn = self.gemini_session.receive()
            # Iterate over responses in the turn
            async for response in turn:
                # Handle audio response
                if hasattr(response, 'data') and response.data:
                    # Try to use audio data directly first
                    if isinstance(response.data, bytes):
                        response_audio += response.data
                        logger.info(f"📥 Received raw audio response from Gemini: {len(response.data)} bytes")
                    elif isinstance(response.data, str):
                        # If it's a string, try to decode as base64
                        try:
                            decoded_audio = base64.b64decode(response.data)
                            response_audio += decoded_audio
                            logger.info(f"📥 Received base64 audio response from Gemini: {len(decoded_audio)} bytes")
                        except Exception as decode_error:
                            logger.warning(f"Could not decode base64 audio: {decode_error}")
                    
                    # Handle text response (for logging)
                    if hasattr(response, 'text') and response.text:
                        self.session_logger.log_transcript(
                            "assistant_response", response.text
                        )
                        logger.info(f"AI Response: {response.text}")
                    
                    # Handle server content (audio in server_content)
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    audio_data = part.inline_data.data
                                    if isinstance(audio_data, bytes):
                                        response_audio += audio_data
                                        logger.info(f"📥 Received inline audio from Gemini: {len(audio_data)} bytes")
                                    elif isinstance(audio_data, str):
                                        try:
                                            decoded_audio = base64.b64decode(audio_data)
                                            response_audio += decoded_audio
                                            logger.info(f"📥 Received base64 inline audio from Gemini: {len(decoded_audio)} bytes")
                                        except:
                                            pass
                    
                    # Break after receiving first response to avoid hanging
                    # In a real conversation, there might be multiple responses, but for now we'll take the first
                    if response_audio:
                        break
        
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            raise  # Re-raise to be handled by caller
        
        return response_audio
    
    async def process_audio(self, audio_data: bytes) -> bytes:
        """Process incoming audio through the voice agent with improved error handling"""
        try:
            if not self.gemini_session:
                logger.info("Voice session not initialized, attempting to initialize...")
                if not await self.initialize_voice_session():
                    logger.error("Failed to initialize voice session")
                    return b""
            
            # Convert telephony audio to Gemini format if needed
            processed_audio = self.convert_telephony_to_gemini(audio_data)
            logger.info(f"🎙️ Processing audio chunk: {len(audio_data)} bytes → Gemini: {len(processed_audio)} bytes")
            
            # Send audio to Gemini with connection error handling
            send_success = False
            for attempt in range(2):  # Try twice
                try:
                    # Send audio in the correct format for Gemini Live API
                    await self.gemini_session.send(
                        input={"data": processed_audio, "mime_type": "audio/pcm"}
                    )
                    send_success = True
                    break
                except Exception as send_error:
                    error_str = str(send_error)
                    logger.error(f"Error sending audio to Gemini (attempt {attempt + 1}): {send_error}")
                    
                    # Handle specific WebSocket errors
                    if "ConnectionClosedOK" in error_str or "ConnectionClosed" in error_str:
                        logger.warning("🔌 WebSocket connection closed, attempting to reconnect...")
                    elif "ConnectionResetError" in error_str:
                        logger.warning("🔌 Connection reset by server, attempting to reconnect...")
                    elif "TimeoutError" in error_str:
                        logger.warning("⏱️ Connection timeout, attempting to reconnect...")
                    else:
                        logger.warning(f"🔧 Unexpected error type: {type(send_error).__name__}")
                    
                    # Try to reinitialize the session
                    self.voice_session = None
                    self.gemini_session = None
                    if attempt == 0:  # Only retry once
                        logger.info("Attempting to reinitialize voice session...")
                        if await self.initialize_voice_session():
                            logger.info("✅ Voice session reinitialized, retrying audio send...")
                        else:
                            logger.error("❌ Failed to reinitialize voice session")
                            return b""
                    else:
                        logger.error("❌ Failed to send audio after retry")
                        return b""
            
            if not send_success:
                logger.error("❌ Unable to send audio to Gemini after retries")
                return b""
            
            # Get response with timeout and better error handling
            response_audio = b""
            response_timeout = 2.0  # Reduced timeout for faster response
            
            try:
                # Receive response without timeout for continuous streaming
                response_audio = await self._receive_response()
                    
            except Exception as receive_error:
                error_str = str(receive_error)
                logger.error(f"Error receiving response from Gemini: {receive_error}")
                
                # Handle specific errors
                if "ConnectionClosedOK" in error_str or "ConnectionClosed" in error_str:
                    logger.warning("🔌 WebSocket connection closed while receiving response")
                elif "ConnectionResetError" in error_str:
                    logger.warning("🔌 Connection reset while receiving response")
                else:
                    logger.warning(f"🔧 Unexpected receive error: {type(receive_error).__name__}")
                
                # Close the session to force reinitialize on next call
                self.voice_session = None
                self.gemini_session = None
                return b""
            
            # Convert back to telephony format
            if response_audio:
                telephony_audio = self.convert_gemini_to_telephony(response_audio)
                logger.info(f"🔊 Converted to telephony format: {len(telephony_audio)} bytes - sending back via RTP")
                return telephony_audio
            else:
                logger.debug("No audio response received from Gemini")
                return b""
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Clear the session to force reinitialize
            self.voice_session = None
            self.gemini_session = None
            return b""
    
    def convert_telephony_to_gemini(self, audio_data: bytes) -> bytes:
        """Convert 8kHz telephony audio to 16kHz for Gemini with better quality"""
        try:
            # Ensure SEND_SAMPLE_RATE is 16000
            target_rate = 16000  # Gemini expects 16kHz for input
            
            logger.debug(f"Converting {len(audio_data)} bytes from 8kHz to {target_rate}Hz")
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=8000,  # 8kHz input
                channels=1       # Mono
            )
            
            # Normalize first to ensure good signal level
            audio = audio.normalize()
            
            # Apply slight volume boost to improve clarity
            audio = audio + 3  # +3 dB boost
            
            # Convert to Gemini's expected sample rate (16kHz)
            audio_16k = audio.set_frame_rate(target_rate)
            
            converted_data = audio_16k.raw_data
            logger.debug(f"Converted to {len(converted_data)} bytes at {target_rate}Hz (ratio: {len(converted_data)/len(audio_data):.1f}x)")
            return converted_data
        except Exception as e:
            logger.error(f"Error converting telephony to Gemini audio: {e}")
            return audio_data
    
    def convert_gemini_to_telephony(self, audio_data: bytes) -> bytes:
        """Convert 24kHz Gemini audio to 8kHz for telephony with advanced audio processing"""
        try:
            # Gemini outputs at 24kHz
            source_rate = 24000
            target_rate = 8000
            
            logger.debug(f"Converting {len(audio_data)} bytes from {source_rate}Hz to {target_rate}Hz")
            
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=source_rate,  # Gemini's output rate
                channels=1       # Mono
            )
            
            # Step 1: Apply pre-emphasis to boost high frequencies before downsampling
            # This compensates for the loss of clarity in telephony systems
            from pydub.effects import high_pass_filter
            audio = high_pass_filter(audio, 50)  # Remove subsonic frequencies
            
            # Step 2: Apply dynamic range compression for consistent levels
            audio = self._apply_dynamic_compression(audio)
            
            # Step 3: Apply anti-aliasing filter with optimal cutoff
            # Use a Butterworth filter approximation for smoother rolloff
            audio = audio.low_pass_filter(3500)  # Slightly higher for voice clarity
            
            # Step 4: Convert to 8kHz using high-quality resampling
            audio_8k = audio.set_frame_rate(target_rate)
            
            # Step 5: Apply telephony band-pass filter with optimized parameters
            from pydub.effects import band_pass_filter
            audio_8k = band_pass_filter(audio_8k, 300, 3400)
            
            # Step 6: Apply automatic gain control (AGC)
            audio_8k = self._apply_agc(audio_8k)
            
            # Step 7: Apply de-emphasis to restore natural sound
            # Compensate for telephony system emphasis
            audio_8k = audio_8k.low_pass_filter(3000)
            
            # Step 8: Apply crossfade if we have previous audio
            if hasattr(self, '_last_audio_tail') and self._last_audio_tail:
                audio_8k = self._apply_crossfade(audio_8k, self._last_audio_tail)
            
            # Store tail for next crossfade
            if len(audio_8k) > 10:  # Store last 10ms for crossfade
                tail_samples = int(0.01 * target_rate * 2)  # 10ms of 16-bit samples
                self._last_audio_tail = audio_8k.raw_data[-tail_samples:]
            
            # Step 9: Apply final limiting to prevent any clipping
            if audio_8k.max_dBFS > -1:
                audio_8k = audio_8k.apply_gain(-audio_8k.max_dBFS - 1)
            
            converted_data = audio_8k.raw_data
            logger.debug(f"Converted to {len(converted_data)} bytes at {target_rate}Hz (ratio: {len(audio_data)/len(converted_data):.1f}x)")
            
            return converted_data
            
        except Exception as e:
            logger.error(f"Error converting Gemini to telephony audio: {e}")
            # Fallback to simple conversion
            try:
                audio = AudioSegment(
                    data=audio_data,
                    sample_width=2,
                    frame_rate=24000,  # Hardcode the rate
                    channels=1
                )
                audio_8k = audio.set_frame_rate(8000)
                return audio_8k.raw_data
            except:
                return audio_data
    
    def _apply_dynamic_compression(self, audio: AudioSegment, ratio=4.0, threshold=-20.0) -> AudioSegment:
        """Apply dynamic range compression to audio"""
        try:
            # Simple compressor implementation
            if audio.dBFS > threshold:
                # Calculate gain reduction
                excess_db = audio.dBFS - threshold
                gain_reduction = excess_db * (1 - 1/ratio)
                audio = audio.apply_gain(-gain_reduction)
            
            # Apply makeup gain to restore average level
            makeup_gain = 3.0  # 3dB makeup gain
            audio = audio.apply_gain(makeup_gain)
            
            return audio
        except Exception as e:
            logger.error(f"Error in dynamic compression: {e}")
            return audio
    
    def _apply_agc(self, audio: AudioSegment, target_level=-18.0) -> AudioSegment:
        """Apply automatic gain control to maintain consistent levels"""
        try:
            # Calculate current RMS level
            current_level = audio.dBFS
            
            # Calculate gain adjustment needed
            gain_adjustment = target_level - current_level
            
            # Limit gain adjustment to prevent extreme changes
            gain_adjustment = max(-12, min(12, gain_adjustment))
            
            # Apply gain with soft knee to prevent artifacts
            if abs(gain_adjustment) > 0.5:
                audio = audio.apply_gain(gain_adjustment * 0.8)  # Apply 80% of needed gain
            
            return audio
        except Exception as e:
            logger.error(f"Error in AGC: {e}")
            return audio
    
    def _apply_crossfade(self, audio: AudioSegment, previous_tail: bytes, crossfade_ms=5) -> AudioSegment:
        """Apply crossfade between audio chunks to eliminate clicks"""
        try:
            # Convert tail to AudioSegment
            tail_segment = AudioSegment(
                data=previous_tail,
                sample_width=2,
                frame_rate=8000,
                channels=1
            )
            
            # Get the overlapping portion from the new audio
            overlap_samples = int(crossfade_ms * 8000 / 1000)
            
            if len(audio) > overlap_samples and len(tail_segment) > 0:
                # Apply linear crossfade
                # Fade out the tail
                tail_faded = tail_segment.fade_out(crossfade_ms)
                
                # Fade in the beginning of new audio
                head = audio[:crossfade_ms]
                head_faded = head.fade_in(crossfade_ms)
                
                # Mix the faded sections
                mixed_head = tail_faded.overlay(head_faded)
                
                # Combine with the rest of the audio
                result = mixed_head + audio[crossfade_ms:]
                
                return result
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Error in crossfade: {e}")
            return audio
    
    def cleanup(self):
        """Clean up the session synchronously"""
        try:
            # Simply clear the sessions without trying to close them async
            # The WebSocket will be closed when the object is garbage collected
            self.voice_session = None
            self.gemini_session = None
            
            self.session_logger.log_transcript("system", "Call ended")
            self.session_logger.save_session()
            self.session_logger.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def _async_close_session(self):
        """Helper to close the voice session asynchronously"""
        if self.voice_session:
            try:
                # Exit the context manager properly
                await self.voice_session.__aexit__(None, None, None)
                logger.info("✅ Voice session context closed properly")
            except Exception as e:
                logger.warning(f"Error closing voice session: {e}")
            finally:
                self.voice_session = None
                self.gemini_session = None
    
    async def async_cleanup(self):
        """Clean up the session asynchronously"""
        try:
            await self._async_close_session()
            
            self.session_logger.log_transcript("system", "Call ended")
            self.session_logger.save_session()
            self.session_logger.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

# Initialize SIP handler
sip_handler = WindowsSIPHandler()

@app.on_event("startup")
async def startup_event():
    """Start SIP listener when FastAPI starts"""
    sip_handler.start_sip_listener()
    logger.info("Windows VoIP Voice Agent started")
    logger.info(f"Phone number: {config['phone_number']}")
    logger.info(f"Local IP: {config['local_ip']}")
    logger.info(f"Gate VoIP: {config['host']}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when shutting down"""
    sip_handler.stop()

@app.post("/api/make_call")
async def make_outbound_call(call_request: dict):
    """API endpoint to initiate an outbound call"""
    try:
        phone_number = call_request.get("phone_number")
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="Phone number required")
        
        # Make call through Gate VoIP
        session_id = sip_handler.make_outbound_call(phone_number)
        
        if session_id:
            return {
                "status": "success",
                "session_id": session_id,
                "phone_number": phone_number,
                "message": "Outbound call initiated through Gate VoIP"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initiate call")
        
    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active call sessions"""
    try:
        sessions = []
        for session_id, session_info in active_sessions.items():
            voice_session = session_info["voice_session"]
            sessions.append({
                "session_id": session_id,
                "caller_id": voice_session.caller_id,
                "called_number": voice_session.called_number,
                "start_time": voice_session.start_time.isoformat(),
                "status": session_info["status"],
                "duration_seconds": (datetime.now(timezone.utc) - session_info["call_start"]).total_seconds()
            })
        
        return {
            "status": "success",
            "active_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "phone_number": config['phone_number'],
        "local_ip": config['local_ip'],
        "gate_voip": config['host'],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "status": "success",
        "config": {
            "phone_number": config['phone_number'],
            "local_ip": config['local_ip'],
            "gate_voip_ip": config['host'],
            "sip_port": config['sip_port'],
            "voice_agent": config['context']
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows VoIP Voice Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Windows VoIP Voice Agent Starting")
    logger.info("=" * 60)
    logger.info(f"Phone Number: {config['phone_number']}")
    logger.info(f"Local IP: {config['local_ip']}")
    logger.info(f"Gate VoIP IP: {config['host']}")
    logger.info(f"SIP Port: {config['sip_port']}")
    logger.info(f"API Server: {args.host}:{args.port}")
    logger.info("=" * 60)
    logger.info("Endpoints:")
    logger.info("  GET /health - System health check")
    logger.info("  GET /api/config - Current configuration")
    logger.info("  GET /api/sessions - Active call sessions")
    logger.info("  POST /api/make_call - Initiate outbound call")
    logger.info("=" * 60)
    
    uvicorn.run(
        "windows_voice_agent:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
