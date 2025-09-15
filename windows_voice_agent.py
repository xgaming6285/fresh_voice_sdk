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
                "text": "You are a helpful AI voice assistant answering phone calls. Keep responses conversational and concise since this is a voice-only interaction."
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
                    for session_id, rtp_session in self.sessions.items():
                        if rtp_session.remote_addr == addr:
                            rtp_session.process_incoming_audio(audio_payload, payload_type, timestamp)
                            break
                
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
            
            # Start processing if not already running
            if not self.processing:
                self.processing = True
                threading.Thread(target=self._process_audio_queue, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
    
    def _process_audio_queue(self):
        """Process queued audio through voice session"""
        audio_buffer = b""
        chunk_size = 1600  # 100ms at 8kHz, 16-bit = 1600 bytes
        
        try:
            while self.processing:
                try:
                    # Get audio chunk with timeout
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer += audio_chunk
                    
                    # Process when we have enough audio
                    if len(audio_buffer) >= chunk_size:
                        chunk_to_process = audio_buffer[:chunk_size]
                        audio_buffer = audio_buffer[chunk_size:]
                        
                        # Process through voice session asynchronously
                        asyncio.create_task(self._process_chunk(chunk_to_process))
                        
                except queue.Empty:
                    # No audio for 100ms, continue
                    if len(audio_buffer) > 0:
                        # Process remaining buffer
                        asyncio.create_task(self._process_chunk(audio_buffer))
                        audio_buffer = b""
                        
                except Exception as e:
                    logger.error(f"Error in audio processing queue: {e}")
                    break
                    
        finally:
            self.processing = False
    
    async def _process_chunk(self, audio_chunk: bytes):
        """Process audio chunk through voice session"""
        try:
            response_audio = await self.voice_session.process_audio(audio_chunk)
            if response_audio:
                self.send_audio(response_audio)
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def send_audio(self, audio_data: bytes):
        """Send audio back via RTP"""
        try:
            # Convert PCM to μ-law for transmission
            ulaw_data = self.pcm_to_ulaw(audio_data)
            
            # Create RTP packet
            rtp_packet = self._create_rtp_packet(ulaw_data, payload_type=0)
            
            # Send packet
            self.rtp_socket.sendto(rtp_packet, self.remote_addr)
            
            logger.debug(f"Sent RTP audio packet: {len(rtp_packet)} bytes")
            
        except Exception as e:
            logger.error(f"Error sending RTP audio: {e}")
    
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
                    asyncio.create_task(voice_session.cleanup())
                    
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
        
        # Log call start
        self.session_logger.log_transcript(
            "system", 
            f"Call started - From: {caller_id}, To: {called_number}"
        )
    
    async def initialize_voice_session(self):
        """Initialize the Google Gemini voice session"""
        try:
            logger.info(f"Attempting to connect to Gemini live session...")
            logger.info(f"Using model: {MODEL}")
            logger.info(f"Voice config: {VOICE_CONFIG}")
            
            self.voice_session = await voice_client.aio.live.connect(
                model=MODEL, 
                config=VOICE_CONFIG
            ).__aenter__()
            
            logger.info(f"✅ Voice session connected successfully for {self.session_id}")
            logger.info(f"📞 Call from {self.caller_id} is now ready for voice interaction")
            
            # Log successful initialization
            self.session_logger.log_transcript(
                "system", 
                f"Voice session initialized - ready for conversation"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize voice session: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def process_audio(self, audio_data: bytes) -> bytes:
        """Process incoming audio through the voice agent"""
        try:
            if not self.voice_session:
                logger.info("Voice session not initialized, attempting to initialize...")
                if not await self.initialize_voice_session():
                    logger.error("Failed to initialize voice session")
                    return b""
            
            # Convert telephony audio to Gemini format if needed
            processed_audio = self.convert_telephony_to_gemini(audio_data)
            logger.debug(f"Sending audio chunk: {len(processed_audio)} bytes")
            
            # Send audio to Gemini
            await self.voice_session.send(
                input={"data": processed_audio, "mime_type": "audio/pcm"}
            )
            
            # Get response
            response_audio = b""
            turn = self.voice_session.receive()
            async for response in turn:
                if hasattr(response, 'data') and response.data:
                    response_audio += response.data
                    logger.debug(f"Received audio response: {len(response.data)} bytes")
                if hasattr(response, 'text') and response.text:
                    self.session_logger.log_transcript(
                        "assistant_response", response.text
                    )
                    logger.info(f"AI Response: {response.text}")
            
            # Convert back to telephony format
            if response_audio:
                telephony_audio = self.convert_gemini_to_telephony(response_audio)
                logger.debug(f"Converted to telephony format: {len(telephony_audio)} bytes")
                return telephony_audio
            else:
                logger.debug("No audio response received")
                return b""
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return b""
    
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
        except:
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
        except:
            return audio_data
    
    async def cleanup(self):
        """Clean up the session"""
        try:
            if self.voice_session:
                await self.voice_session.__aexit__(None, None, None)
            
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
