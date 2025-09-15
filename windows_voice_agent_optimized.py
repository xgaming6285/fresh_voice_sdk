#!/usr/bin/env python3
"""
Optimized Windows Voice Agent with Gemini Live API - Low Latency Version
Reduced buffering and optimized audio processing for faster response times
"""

import os
import sys
import json
import socket
import threading
import asyncio
import queue
import time
import struct
import uuid
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('windows_voice_agent.log')
    ]
)
logger = logging.getLogger("windows_voice_agent")

# Audio processing imports
try:
    from pydub import AudioSegment
    from pydub.utils import which
    # Ensure ffmpeg is available
    AudioSegment.converter = which("ffmpeg")
except ImportError:
    logger.error("pydub not installed. Run: pip install pydub")
    sys.exit(1)

# Google AI imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    logger.error("google-genai not installed. Run: pip install google-generativeai")
    sys.exit(1)

# Audio sample rates
SEND_SAMPLE_RATE = 16000    # Gemini expects 16kHz for input
RECEIVE_SAMPLE_RATE = 24000  # Gemini outputs 24kHz

# Performance optimizations
AUDIO_CHUNK_MS = 40  # Reduced from 80ms to 40ms for lower latency
MIN_AUDIO_BUFFER_MS = 120  # Reduced from 320ms to 120ms
RTP_PACKET_MS = 20  # Standard RTP packet duration
RESPONSE_CHUNK_SIZE = 240  # Reduced chunk size for faster response

# Load configuration
CONFIG_FILE = "asterisk_config.json"
if not os.path.exists(CONFIG_FILE):
    logger.error(f"Configuration file {CONFIG_FILE} not found!")
    sys.exit(1)

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Initialize FastAPI
app = FastAPI(title="Windows Voice Agent")

# Track active sessions
active_sessions: Dict[str, Any] = {}

class SessionLogger:
    """Handles session logging and transcript management"""
    
    def __init__(self, session_id: str, caller_id: str, called_number: str):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        self.transcript = []
        self.session_dir = Path(f"sessions/{session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session metadata
        self.metadata = {
            "session_id": session_id,
            "caller_id": caller_id,
            "called_number": called_number,
            "start_time": self.start_time.isoformat(),
            "agent_context": config.get("context", "Default voice agent")
        }
        
        logger.info(f"📝 Session logger initialized for {session_id}")
    
    def log_transcript(self, speaker: str, text: str):
        """Add entry to transcript"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speaker": speaker,
            "text": text
        }
        self.transcript.append(entry)
        
    def save_session(self):
        """Save session data to disk"""
        try:
            # Update metadata
            self.metadata["end_time"] = datetime.now(timezone.utc).isoformat()
            self.metadata["duration_seconds"] = (
                datetime.now(timezone.utc) - self.start_time
            ).total_seconds()
            
            # Save transcript
            transcript_file = self.session_dir / f"transcript_{self.session_id}.json"
            session_data = {
                "metadata": self.metadata,
                "transcript": self.transcript
            }
            
            with open(transcript_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"✅ Session data saved in: {self.session_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving session data: {e}")
    
    def close(self):
        """Close the session logger"""
        self.save_session()

class RTPSession:
    """Handles RTP audio streaming with performance optimizations"""
    
    def __init__(self, session_id: str, remote_addr: Tuple[str, int], voice_session: 'WindowsVoiceSession'):
        self.session_id = session_id
        self.remote_addr = remote_addr
        self.voice_session = voice_session
        self.processing = True
        
        # Audio queues with smaller buffers
        self.audio_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.output_queue = queue.Queue(maxsize=20)
        
        # RTP parameters
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = struct.unpack('!I', os.urandom(4))[0]
        
        # Audio buffer for accumulation (reduced size)
        self.audio_buffer = b""
        self.min_buffer_size = int(8000 * 2 * MIN_AUDIO_BUFFER_MS / 1000)  # 120ms at 8kHz
        
        # Performance tracking
        self.last_process_time = time.time()
        self.processing_count = 0
        
        # Start async processing with optimized loop
        self.loop = asyncio.new_event_loop()
        self.processing_thread = threading.Thread(
            target=self._run_async_processing,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info(f"🎵 Created optimized RTP session {session_id} for {remote_addr}")
    
    def _run_async_processing(self):
        """Run async processing in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_audio_processing())
    
    def process_incoming_audio(self, audio_data: bytes, payload_type: int, timestamp: int):
        """Process incoming RTP audio with minimal buffering"""
        try:
            # Convert based on payload type
            if payload_type == 0:  # PCMU (μ-law)
                pcm_audio = self.ulaw_to_pcm(audio_data)
            elif payload_type == 8:  # PCMA (A-law)
                pcm_audio = self.alaw_to_pcm(audio_data)
            else:
                pcm_audio = audio_data  # Assume PCM
            
            # Add to queue immediately for low latency
            try:
                self.audio_queue.put_nowait(pcm_audio)
                logger.info(f"🎤 Queued {len(pcm_audio)} bytes of PCM audio for processing")
            except queue.Full:
                logger.warning("Audio queue full, dropping oldest packet")
                try:
                    self.audio_queue.get_nowait()  # Remove oldest
                    self.audio_queue.put_nowait(pcm_audio)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
    
    async def _async_audio_processing(self):
        """Optimized async audio processing with continuous streaming"""
        logger.info("🎧 Starting optimized audio processor")
        
        try:
            # Start continuous response receiver first
            receive_task = asyncio.create_task(self._continuous_receive_responses())
            
            # Process incoming audio with minimal buffering
            while self.processing:
                try:
                    # Accumulate audio chunks quickly
                    accumulated_audio = b""
                    deadline = time.time() + (AUDIO_CHUNK_MS / 1000.0)
                    
                    # Collect audio until deadline or minimum size
                    while time.time() < deadline and len(accumulated_audio) < self.min_buffer_size:
                        try:
                            # Non-blocking get with very short timeout
                            chunk = await asyncio.get_event_loop().run_in_executor(
                                None, self.audio_queue.get, True, 0.005
                            )
                            accumulated_audio += chunk
                        except queue.Empty:
                            if accumulated_audio:
                                break  # Process what we have
                            else:
                                await asyncio.sleep(0.001)  # Very brief sleep
                                continue
                    
                    # Process accumulated audio immediately
                    if accumulated_audio:
                        # Send to Gemini without waiting for response
                        if self.voice_session.gemini_session:
                            await self._send_audio_to_gemini(accumulated_audio)
                        
                except Exception as e:
                    logger.error(f"Error in audio processing: {e}")
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Fatal error in audio processing: {e}")
        finally:
            if 'receive_task' in locals():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
    
    async def _send_audio_to_gemini(self, audio_chunk: bytes):
        """Send audio to Gemini with minimal latency"""
        try:
            if not self.voice_session.gemini_session:
                return
                
            # Convert and send immediately
            processed_audio = self.voice_session.convert_telephony_to_gemini(audio_chunk)
            logger.info(f"🎙️ Sending audio to Gemini: {len(audio_chunk)} bytes → {len(processed_audio)} bytes")
            
            await self.voice_session.gemini_session.send(
                input={"data": processed_audio, "mime_type": "audio/pcm"}
            )
            
            self.processing_count += 1
            
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")
    
    async def _continuous_receive_responses(self):
        """Optimized continuous response receiver"""
        logger.info("🎧 Starting optimized response receiver")
        
        while self.processing and self.voice_session.gemini_session:
            try:
                turn = self.voice_session.gemini_session.receive()
                
                async for response in turn:
                    try:
                        # Process server content first (most common)
                        if hasattr(response, 'server_content') and response.server_content:
                            if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    # Audio data
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        if hasattr(part.inline_data, 'mime_type') and 'audio' in part.inline_data.mime_type:
                                            await self._process_audio_response(part.inline_data.data)
                                    
                                    # Text data for logging
                                    if hasattr(part, 'text') and part.text:
                                        self.voice_session.session_logger.log_transcript(
                                            "assistant", part.text.strip()
                                        )
                                        logger.info(f"AI: {part.text.strip()}")
                        
                        # Legacy format support
                        elif hasattr(response, 'data') and response.data:
                            await self._process_audio_response(response.data)
                            
                    except Exception as e:
                        logger.error(f"Error processing response: {e}")
                        continue
                        
            except Exception as e:
                if self.processing:
                    logger.error(f"Error in response receiver: {e}")
                    await asyncio.sleep(0.05)
    
    async def _process_audio_response(self, audio_data):
        """Process and send audio response with minimal latency"""
        try:
            # Decode if base64
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
                
            logger.info(f"📥 Received {len(audio_bytes)} bytes from Gemini")
            
            # Convert to telephony format
            telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_bytes)
            
            # Send immediately in small chunks without excessive buffering
            chunk_size = RESPONSE_CHUNK_SIZE  # 240 bytes = 30ms at 8kHz
            for i in range(0, len(telephony_audio), chunk_size):
                chunk = telephony_audio[i:i + chunk_size]
                self.send_audio(chunk)
                # Minimal pacing to prevent overflow
                await asyncio.sleep(0.02)  # 20ms between chunks
                
        except Exception as e:
            logger.error(f"Error processing audio response: {e}")
    
    def send_audio(self, audio_data: bytes):
        """Queue audio for immediate RTP transmission"""
        try:
            # Convert PCM to μ-law
            ulaw_data = self.pcm_to_ulaw(audio_data)
            
            # Send immediately if queue is small
            if self.output_queue.qsize() < 5:
                self.output_queue.put_nowait(ulaw_data)
                logger.info(f"📤 Queued {len(ulaw_data)} bytes for RTP transmission")
            else:
                logger.warning("Output queue congested, dropping audio")
                
        except Exception as e:
            logger.error(f"Error queueing audio: {e}")
    
    def _process_output_queue(self):
        """Process output queue with minimal latency"""
        packet_size = 160  # 20ms of μ-law audio at 8kHz
        packet_duration_s = 0.020  # 20ms per packet
        last_packet_time = time.time()
        
        while self.processing:
            try:
                # Get audio from queue quickly
                try:
                    audio_data = self.output_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # Send audio in properly paced packets
                for i in range(0, len(audio_data), packet_size):
                    chunk = audio_data[i:i + packet_size]
                    
                    # Minimal pacing
                    current_time = time.time()
                    time_since_last = current_time - last_packet_time
                    if time_since_last < packet_duration_s:
                        time.sleep(packet_duration_s - time_since_last)
                    
                    # Send packet
                    rtp_packet = self._create_rtp_packet(chunk, payload_type=0)
                    self.rtp_socket.sendto(rtp_packet, self.remote_addr)
                    last_packet_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error in output processing: {e}")
    
    def _create_rtp_packet(self, payload: bytes, payload_type: int = 0):
        """Create RTP packet"""
        version = 2
        padding = 0
        extension = 0
        cc = 0
        marker = 0
        
        byte0 = (version << 6) | (padding << 5) | (extension << 4) | cc
        byte1 = (marker << 7) | payload_type
        
        header = struct.pack('!BBHII', 
                           byte0, byte1, 
                           self.sequence_number, 
                           self.timestamp, 
                           self.ssrc)
        
        self.sequence_number = (self.sequence_number + 1) & 0xFFFF
        self.timestamp = (self.timestamp + len(payload)) & 0xFFFFFFFF
        
        return header + payload
    
    def ulaw_to_pcm(self, ulaw_data: bytes) -> bytes:
        """Convert μ-law to 16-bit PCM"""
        import audioop
        return audioop.ulaw2lin(ulaw_data, 2)
    
    def alaw_to_pcm(self, alaw_data: bytes) -> bytes:
        """Convert A-law to 16-bit PCM"""
        import audioop
        return audioop.alaw2lin(alaw_data, 2)
    
    def pcm_to_ulaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to μ-law"""
        import audioop
        return audioop.lin2ulaw(pcm_data, 2)

class RTPServer:
    """Optimized RTP server for handling multiple sessions"""
    
    def __init__(self, local_ip: str, port_range_start: int = 10000):
        self.local_ip = local_ip
        self.port_range_start = port_range_start
        self.socket = None
        self.port = None
        self.running = False
        self.sessions: Dict[str, RTPSession] = {}
        
    def start(self):
        """Start RTP server with automatic port selection"""
        for port_offset in range(100):
            try:
                port = self.port_range_start + port_offset * 2
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Increase receive buffer for better performance
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                self.socket.bind((self.local_ip, port))
                self.port = port
                self.running = True
                
                # Start listener thread
                self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
                self.listener_thread.start()
                
                # Start output processor for each session
                self.output_thread = threading.Thread(target=self._process_all_outputs, daemon=True)
                self.output_thread.start()
                
                logger.info(f"📡 RTP server started on {self.local_ip}:{port}")
                break
                
            except OSError:
                continue
                
        if not self.running:
            raise RuntimeError("Failed to start RTP server - no available ports")
    
    def _listen_loop(self):
        """Main RTP listening loop with optimized processing"""
        while self.running:
            try:
                # Receive with larger buffer
                data, addr = self.socket.recvfrom(2048)
                
                if len(data) < 12:  # Minimum RTP header size
                    continue
                
                # Parse RTP header quickly
                header = data[:12]
                payload = data[12:]
                
                version = (header[0] >> 6) & 0x03
                payload_type = header[1] & 0x7F
                sequence = struct.unpack('!H', header[2:4])[0]
                timestamp = struct.unpack('!I', header[4:8])[0]
                ssrc = struct.unpack('!I', header[8:12])[0]
                
                logger.info(f"🎵 Received RTP audio from {addr}: {len(payload)} bytes, PT={payload_type}")
                
                # Find session by IP (port may differ)
                session_found = False
                for session_id, rtp_session in self.sessions.items():
                    if rtp_session.remote_addr[0] == addr[0]:
                        # Update port if needed
                        if rtp_session.remote_addr != addr:
                            logger.info(f"📍 Updating RTP address from {rtp_session.remote_addr} to {addr}")
                            rtp_session.remote_addr = addr
                        
                        rtp_session.process_incoming_audio(payload, payload_type, timestamp)
                        session_found = True
                        break
                
                if not session_found:
                    logger.warning(f"⚠️ Received RTP from unknown address: {addr}")
                    
            except Exception as e:
                logger.error(f"Error in RTP listener: {e}")
    
    def _process_all_outputs(self):
        """Process output queues for all sessions"""
        while self.running:
            try:
                for session in list(self.sessions.values()):
                    if session.processing:
                        session._process_output_queue()
                time.sleep(0.001)  # Brief sleep
            except Exception as e:
                logger.error(f"Error in output processor: {e}")
    
    def create_session(self, session_id: str, remote_addr: Tuple[str, int], 
                      voice_session: 'WindowsVoiceSession') -> RTPSession:
        """Create new RTP session"""
        session = RTPSession(session_id, remote_addr, voice_session)
        session.rtp_socket = self.socket  # Share socket
        self.sessions[session_id] = session
        
        # Start output processing thread
        output_thread = threading.Thread(
            target=session._process_output_queue,
            daemon=True
        )
        output_thread.start()
        
        return session
    
    def remove_session(self, session_id: str):
        """Remove RTP session"""
        if session_id in self.sessions:
            self.sessions[session_id].processing = False
            del self.sessions[session_id]
            logger.info(f"🎵 Removed RTP session {session_id}")
    
    def stop(self):
        """Stop RTP server"""
        self.running = False
        if self.socket:
            self.socket.close()

class WindowsVoiceSession:
    """Optimized voice session with Gemini Live API"""
    
    def __init__(self, session_id: str, caller_id: str, called_number: str):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        
        # Initialize session logger
        self.session_logger = SessionLogger(session_id, caller_id, called_number)
        
        # Google AI setup
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.voice_session = None
        self.gemini_session = None
        
        # Connection retry settings
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.connection_backoff = 1.0
        
        logger.info(f"🎤 Created optimized voice session {session_id} for {caller_id}")
    
    async def initialize_voice_session(self) -> bool:
        """Initialize Gemini voice session with optimizations"""
        try:
            self.connection_attempts += 1
            
            if self.connection_attempts > self.max_connection_attempts:
                logger.error(f"❌ Max connection attempts reached ({self.max_connection_attempts})")
                return False
            
            logger.info(f"🔌 Initializing Gemini voice session (attempt {self.connection_attempts})...")
            
            # Create voice session with optimized config
            self.voice_session = self.client.aio.live.connect(
                model="models/gemini-2.0-flash-exp",
                config={
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Puck"  # Fast, clear voice
                            }
                        }
                    }
                }
            )
            
            self.gemini_session = await self.voice_session.__aenter__()
            logger.info("✅ Gemini voice session initialized successfully")
            
            # Send context immediately
            context_message = config.get('context', 'You are a helpful voice assistant.')
            await self.gemini_session.send(
                text=f"You are in a phone conversation. {context_message} "
                     f"Respond naturally and concisely. Keep responses brief and conversational. "
                     f"The caller's number is {self.caller_id}."
            )
            
            # Trigger initial greeting with minimal audio
            await asyncio.sleep(0.1)
            
            # Send very short noise burst to trigger response
            noise_samples = int(SEND_SAMPLE_RATE * 0.1)  # 100ms
            noise_audio = bytes([0] * (noise_samples * 2))  # Silence
            
            await self.gemini_session.send(
                input={"data": noise_audio, "mime_type": "audio/pcm"}
            )
            
            # Send greeting instruction
            await self.gemini_session.send(
                text="Start with a brief greeting like 'Hello, how can I help you?'"
            )
            
            logger.info("🎤 Voice session ready for conversation")
            
            self.session_logger.log_transcript(
                "system", 
                f"Voice session initialized - ready for conversation"
            )
            
            # Reset connection tracking
            self.connection_attempts = 0
            self.connection_backoff = 1.0
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize voice session: {e}")
            self.voice_session = None
            self.gemini_session = None
            self.connection_backoff = min(self.connection_backoff * 2, 30.0)
            return False
    
    def convert_telephony_to_gemini(self, audio_data: bytes) -> bytes:
        """Convert 8kHz telephony audio to 16kHz for Gemini"""
        try:
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=8000,
                channels=1
            )
            audio_16k = audio.set_frame_rate(SEND_SAMPLE_RATE)
            return audio_16k.raw_data
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return audio_data
    
    def convert_gemini_to_telephony(self, audio_data: bytes) -> bytes:
        """Convert 24kHz Gemini audio to 8kHz for telephony"""
        try:
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=RECEIVE_SAMPLE_RATE,
                channels=1
            )
            audio_8k = audio.set_frame_rate(8000)
            return audio_8k.raw_data
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return audio_data
    
    async def async_cleanup(self):
        """Clean up session"""
        try:
            if self.voice_session:
                await self.voice_session.__aexit__(None, None, None)
            
            self.session_logger.log_transcript("system", "Call ended")
            self.session_logger.save_session()
            self.session_logger.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

class WindowsSIPHandler:
    """SIP handler for Windows - interfaces with Gate VoIP system"""
    
    def __init__(self):
        self.config = config
        self.local_ip = self.config['local_ip']
        self.gate_ip = self.config['host']
        self.sip_port = self.config['sip_port']
        self.phone_number = self.config['phone_number']
        self.running = False
        self.socket = None
        
        # Initialize RTP server
        self.rtp_server = RTPServer(self.local_ip)
        
    def start_sip_listener(self):
        """Start listening for SIP messages"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.local_ip, self.sip_port))
            self.running = True
            
            logger.info(f"SIP listener started on {self.local_ip}:{self.sip_port}")
            
            # Start RTP server
            self.rtp_server.start()
            
            # Start listening thread
            threading.Thread(target=self._listen_loop, daemon=True).start()
            
            logger.info("🎯 Ready to receive calls from Gate VoIP")
            
        except Exception as e:
            logger.error(f"Failed to start SIP listener: {e}")
    
    def _listen_loop(self):
        """Main SIP listening loop"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = data.decode('utf-8')
                
                # Handle different message types
                first_line = message.split('\n')[0].strip()
                
                if first_line.startswith('INVITE'):
                    self._handle_invite(message, addr)
                elif first_line.startswith('OPTIONS'):
                    self._handle_options(message, addr)
                elif first_line.startswith('BYE'):
                    self._handle_bye(message, addr)
                elif first_line.startswith('ACK'):
                    pass  # ACK doesn't need response
                    
            except Exception as e:
                logger.error(f"Error in SIP listener: {e}")
    
    def _handle_invite(self, message: str, addr):
        """Handle incoming INVITE"""
        try:
            logger.info("📞 ================== INCOMING CALL ==================")
            
            # Parse caller info
            caller_id = "Unknown"
            call_id = ""
            
            for line in message.split('\n'):
                line = line.strip()
                if line.startswith('From:'):
                    if '<sip:' in line:
                        start = line.find('<sip:') + 5
                        end = line.find('@', start)
                        if end > start:
                            caller_id = line[start:end]
                elif line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
            
            logger.info(f"📞 Incoming call from {caller_id}")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Send 180 Ringing
            ringing_response = self._create_sip_ringing_response(message, session_id)
            self.socket.sendto(ringing_response.encode(), addr)
            logger.info("🔔 Sent 180 Ringing")
            
            # Brief delay
            time.sleep(0.5)
            
            # Create voice session
            voice_session = WindowsVoiceSession(session_id, caller_id, self.phone_number)
            
            # Initialize voice session in background
            init_task = asyncio.create_task(voice_session.initialize_voice_session())
            
            # Create RTP session
            rtp_session = self.rtp_server.create_session(session_id, addr, voice_session)
            
            # Send 200 OK with SDP
            ok_response = self._create_sip_ok_response(message, session_id)
            self.socket.sendto(ok_response.encode(), addr)
            logger.info(f"✅ Sent 200 OK - RTP port: {self.rtp_server.port}")
            
            # Store session
            active_sessions[session_id] = {
                "voice_session": voice_session,
                "rtp_session": rtp_session,
                "sip_addr": addr,
                "call_id": call_id,
                "status": "active",
                "call_start": datetime.now(timezone.utc)
            }
            
            logger.info(f"☎️ Call established - Session ID: {session_id}")
            
        except Exception as e:
            logger.error(f"Error handling INVITE: {e}")
    
    def _handle_options(self, message: str, addr):
        """Handle OPTIONS keep-alive"""
        try:
            response = "SIP/2.0 200 OK\r\n"
            response += f"Via: {self._extract_via(message)}\r\n"
            response += f"From: {self._extract_from(message)}\r\n"
            response += f"To: {self._extract_to(message)}\r\n"
            response += f"Call-ID: {self._extract_call_id(message)}\r\n"
            response += f"CSeq: {self._extract_cseq(message)}\r\n"
            response += "Content-Length: 0\r\n\r\n"
            
            self.socket.sendto(response.encode(), addr)
            logger.info(f"✅ Responded to OPTIONS from {addr}")
            
        except Exception as e:
            logger.error(f"Error handling OPTIONS: {e}")
    
    def _handle_bye(self, message: str, addr):
        """Handle BYE (call end)"""
        try:
            call_id = self._extract_call_id(message)
            
            # Find and end session
            session_to_end = None
            for session_id, session_info in active_sessions.items():
                if session_info["call_id"] == call_id:
                    session_to_end = session_id
                    break
            
            if session_to_end:
                # Clean up session
                session_info = active_sessions[session_to_end]
                
                # Stop RTP session
                self.rtp_server.remove_session(session_to_end)
                
                # Clean up voice session
                voice_session = session_info["voice_session"]
                asyncio.create_task(voice_session.async_cleanup())
                
                # Remove from active sessions
                del active_sessions[session_to_end]
                
                logger.info(f"Call ended: {session_to_end}")
            
            # Send 200 OK response
            response = "SIP/2.0 200 OK\r\n"
            response += f"Via: {self._extract_via(message)}\r\n"
            response += f"From: {self._extract_from(message)}\r\n"
            response += f"To: {self._extract_to(message)}\r\n"
            response += f"Call-ID: {call_id}\r\n"
            response += f"CSeq: {self._extract_cseq(message)}\r\n"
            response += "Content-Length: 0\r\n\r\n"
            
            self.socket.sendto(response.encode(), addr)
            
        except Exception as e:
            logger.error(f"Error handling BYE: {e}")
    
    def _create_sip_ringing_response(self, invite_message: str, session_id: str) -> str:
        """Create 180 Ringing response"""
        response = "SIP/2.0 180 Ringing\r\n"
        response += f"Via: {self._extract_via(invite_message)}\r\n"
        response += f"From: {self._extract_from(invite_message)}\r\n"
        response += f"To: {self._extract_to(invite_message)};tag={session_id[:8]}\r\n"
        response += f"Call-ID: {self._extract_call_id(invite_message)}\r\n"
        response += f"CSeq: {self._extract_cseq(invite_message)}\r\n"
        response += f"Contact: <sip:{self.phone_number}@{self.local_ip}:{self.sip_port}>\r\n"
        response += "Content-Length: 0\r\n\r\n"
        return response
    
    def _create_sip_ok_response(self, invite_message: str, session_id: str) -> str:
        """Create 200 OK response with SDP"""
        sdp_body = f"v=0\r\n"
        sdp_body += f"o=- {int(time.time())} {int(time.time())} IN IP4 {self.local_ip}\r\n"
        sdp_body += f"s=VoiceAgent\r\n"
        sdp_body += f"c=IN IP4 {self.local_ip}\r\n"
        sdp_body += f"t=0 0\r\n"
        sdp_body += f"m=audio {self.rtp_server.port} RTP/AVP 0 8 101\r\n"
        sdp_body += f"a=rtpmap:0 PCMU/8000\r\n"
        sdp_body += f"a=rtpmap:8 PCMA/8000\r\n"
        sdp_body += f"a=rtpmap:101 telephone-event/8000\r\n"
        sdp_body += f"a=sendrecv\r\n"
        sdp_body += f"a=ptime:20\r\n"
        
        response = "SIP/2.0 200 OK\r\n"
        response += f"Via: {self._extract_via(invite_message)}\r\n"
        response += f"From: {self._extract_from(invite_message)}\r\n"
        response += f"To: {self._extract_to(invite_message)};tag={session_id[:8]}\r\n"
        response += f"Call-ID: {self._extract_call_id(invite_message)}\r\n"
        response += f"CSeq: {self._extract_cseq(invite_message)}\r\n"
        response += f"Contact: <sip:{self.phone_number}@{self.local_ip}:{self.sip_port}>\r\n"
        response += "Content-Type: application/sdp\r\n"
        response += f"Content-Length: {len(sdp_body)}\r\n\r\n"
        response += sdp_body
        
        return response
    
    def _extract_via(self, message: str) -> str:
        for line in message.split('\n'):
            if line.strip().startswith('Via:'):
                return line.split(':', 1)[1].strip()
        return ""
    
    def _extract_from(self, message: str) -> str:
        for line in message.split('\n'):
            if line.strip().startswith('From:'):
                return line.split(':', 1)[1].strip()
        return ""
    
    def _extract_to(self, message: str) -> str:
        for line in message.split('\n'):
            if line.strip().startswith('To:'):
                return line.split(':', 1)[1].strip()
        return ""
    
    def _extract_call_id(self, message: str) -> str:
        for line in message.split('\n'):
            if line.strip().startswith('Call-ID:'):
                return line.split(':', 1)[1].strip()
        return ""
    
    def _extract_cseq(self, message: str) -> str:
        for line in message.split('\n'):
            if line.strip().startswith('CSeq:'):
                return line.split(':', 1)[1].strip()
        return ""
    
    def stop(self):
        """Stop SIP handler"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.rtp_server.stop()

# Initialize SIP handler
sip_handler = WindowsSIPHandler()

@app.on_event("startup")
async def startup_event():
    """Start SIP listener when FastAPI starts"""
    sip_handler.start_sip_listener()
    logger.info("🚀 Optimized Windows Voice Agent started")
    logger.info(f"📞 Phone: {config['phone_number']}")
    logger.info(f"🌐 Local: {config['local_ip']}:{config['sip_port']}")
    logger.info(f"🎯 Gate: {config['host']}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when shutting down"""
    sip_handler.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "optimizations": {
            "audio_chunk_ms": AUDIO_CHUNK_MS,
            "min_buffer_ms": MIN_AUDIO_BUFFER_MS,
            "response_chunk_size": RESPONSE_CHUNK_SIZE
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Windows Voice Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 Optimized Windows Voice Agent Starting")
    logger.info("=" * 60)
    logger.info(f"📞 Phone Number: {config['phone_number']}")
    logger.info(f"🌐 Local IP: {config['local_ip']}")
    logger.info(f"🎯 Gate VoIP: {config['host']}")
    logger.info(f"⚡ Performance Optimizations:")
    logger.info(f"   - Audio chunk: {AUDIO_CHUNK_MS}ms")
    logger.info(f"   - Min buffer: {MIN_AUDIO_BUFFER_MS}ms")
    logger.info(f"   - Response chunk: {RESPONSE_CHUNK_SIZE} bytes")
    logger.info("=" * 60)
    
    uvicorn.run(
        "windows_voice_agent_optimized:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
