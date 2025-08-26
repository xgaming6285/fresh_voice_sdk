# -*- coding: utf-8 -*-
"""
Asterisk Integration Layer for Voice Agent
Handles SIP connections and call routing for the Google Gemini voice agent
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import wave
import io

# SIP and telephony libraries
import pjsua2 as pj
from pydub import AudioSegment
import requests
from websockets import connect, ConnectionClosed

# Audio processing
import pyaudio
import audioop

from main import SessionLogger, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, FORMAT, CHANNELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AsteriskConfig:
    """Configuration for Asterisk connection"""
    host: str
    username: str
    password: str
    sip_port: int = 5060
    ari_port: int = 8088
    ari_username: str = "asterisk"
    ari_password: str = "asterisk"
    context: str = "voice-agent"

@dataclass
class CallSession:
    """Represents an active phone call session"""
    call_id: str
    caller_id: str
    call_direction: str  # "inbound" or "outbound"
    start_time: datetime
    voice_agent_session: Optional[Any] = None
    session_logger: Optional[SessionLogger] = None
    
class AudioConverter:
    """Handles audio format conversion between telephony and Google Gemini formats"""
    
    # Telephony typically uses 8kHz μ-law or A-law
    TELEPHONY_RATE = 8000
    TELEPHONY_FORMAT = pyaudio.paInt16
    
    @staticmethod
    def telephony_to_gemini(audio_data: bytes, input_rate: int = TELEPHONY_RATE) -> bytes:
        """Convert telephony audio (8kHz) to Gemini format (16kHz)"""
        try:
            # Convert to AudioSegment for easy manipulation
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=input_rate,
                channels=1
            )
            
            # Resample to 16kHz for Gemini input
            audio_16k = audio.set_frame_rate(SEND_SAMPLE_RATE)
            return audio_16k.raw_data
            
        except Exception as e:
            logger.error(f"Error converting telephony to Gemini audio: {e}")
            return audio_data
    
    @staticmethod
    def gemini_to_telephony(audio_data: bytes, output_rate: int = TELEPHONY_RATE) -> bytes:
        """Convert Gemini audio (24kHz) to telephony format (8kHz)"""
        try:
            # Gemini outputs 24kHz
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=RECEIVE_SAMPLE_RATE,
                channels=1
            )
            
            # Resample to telephony rate (usually 8kHz)
            audio_8k = audio.set_frame_rate(output_rate)
            return audio_8k.raw_data
            
        except Exception as e:
            logger.error(f"Error converting Gemini to telephony audio: {e}")
            return audio_data

class AsteriskARIClient:
    """Asterisk REST Interface client for call control"""
    
    def __init__(self, config: AsteriskConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.ari_port}/ari"
        self.session = requests.Session()
        self.session.auth = (config.ari_username, config.ari_password)
        
    async def connect_websocket(self):
        """Connect to ARI WebSocket for real-time events"""
        ws_url = f"ws://{self.config.host}:{self.config.ari_port}/ari/events"
        ws_url += f"?api_key={self.config.ari_username}:{self.config.ari_password}&app=voice-agent"
        
        self.websocket = await connect(ws_url)
        logger.info("Connected to Asterisk ARI WebSocket")
        
    async def listen_for_events(self, event_handler: Callable):
        """Listen for ARI events and handle them"""
        try:
            async for message in self.websocket:
                event = json.loads(message)
                await event_handler(event)
        except ConnectionClosed:
            logger.warning("ARI WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in ARI event listener: {e}")
    
    def answer_call(self, channel_id: str):
        """Answer an incoming call"""
        try:
            response = self.session.post(f"{self.base_url}/channels/{channel_id}/answer")
            response.raise_for_status()
            logger.info(f"Answered call on channel {channel_id}")
            return True
        except Exception as e:
            logger.error(f"Error answering call {channel_id}: {e}")
            return False
    
    def hangup_call(self, channel_id: str):
        """Hang up a call"""
        try:
            response = self.session.delete(f"{self.base_url}/channels/{channel_id}")
            response.raise_for_status()
            logger.info(f"Hung up call on channel {channel_id}")
            return True
        except Exception as e:
            logger.error(f"Error hanging up call {channel_id}: {e}")
            return False
    
    def originate_call(self, endpoint: str, caller_id: str = None):
        """Originate an outbound call"""
        try:
            data = {
                "endpoint": endpoint,
                "app": "voice-agent",
                "appArgs": "outbound"
            }
            if caller_id:
                data["callerId"] = caller_id
                
            response = self.session.post(f"{self.base_url}/channels", json=data)
            response.raise_for_status()
            
            channel_info = response.json()
            logger.info(f"Originated call to {endpoint}, channel: {channel_info['id']}")
            return channel_info
            
        except Exception as e:
            logger.error(f"Error originating call to {endpoint}: {e}")
            return None

class TelephonyVoiceAgent:
    """Main class that integrates the voice agent with telephony"""
    
    def __init__(self, config: AsteriskConfig):
        self.config = config
        self.ari_client = AsteriskARIClient(config)
        self.active_calls: Dict[str, CallSession] = {}
        self.audio_converter = AudioConverter()
        
        # Import the voice agent components
        from main import AudioLoop, client, MODEL, CONFIG
        self.voice_client = client
        self.model = MODEL
        self.voice_config = CONFIG
        
    async def start(self):
        """Start the telephony voice agent"""
        logger.info("Starting Telephony Voice Agent")
        
        # Connect to Asterisk ARI
        await self.ari_client.connect_websocket()
        
        # Start listening for telephony events
        event_task = asyncio.create_task(
            self.ari_client.listen_for_events(self.handle_ari_event)
        )
        
        logger.info("Telephony Voice Agent is ready for calls")
        
        try:
            await event_task
        except KeyboardInterrupt:
            logger.info("Shutting down Telephony Voice Agent")
        finally:
            # Clean up active calls
            for call_id, call_session in self.active_calls.items():
                await self.hangup_call(call_id)
    
    async def handle_ari_event(self, event: Dict[str, Any]):
        """Handle events from Asterisk ARI"""
        event_type = event.get("type")
        
        if event_type == "StasisStart":
            # New call entered our application
            await self.handle_new_call(event)
            
        elif event_type == "StasisEnd":
            # Call left our application
            await self.handle_call_end(event)
            
        elif event_type == "ChannelDtmfReceived":
            # DTMF tone received
            await self.handle_dtmf(event)
            
        logger.debug(f"Received ARI event: {event_type}")
    
    async def handle_new_call(self, event: Dict[str, Any]):
        """Handle a new incoming or outgoing call"""
        channel = event["channel"]
        channel_id = channel["id"]
        caller_id = channel.get("caller", {}).get("number", "Unknown")
        
        logger.info(f"New call from {caller_id} on channel {channel_id}")
        
        # Answer the call
        if self.ari_client.answer_call(channel_id):
            # Create call session
            call_session = CallSession(
                call_id=channel_id,
                caller_id=caller_id,
                call_direction="inbound",
                start_time=datetime.now(timezone.utc)
            )
            
            # Initialize session logger for this call
            call_session.session_logger = SessionLogger()
            call_session.session_logger.log_transcript(
                "system", f"Call started from {caller_id}"
            )
            
            self.active_calls[channel_id] = call_session
            
            # Start voice agent session for this call
            await self.start_voice_session(channel_id)
    
    async def handle_call_end(self, event: Dict[str, Any]):
        """Handle call termination"""
        channel = event["channel"]
        channel_id = channel["id"]
        
        if channel_id in self.active_calls:
            call_session = self.active_calls[channel_id]
            
            # End voice agent session
            if call_session.voice_agent_session:
                try:
                    await call_session.voice_agent_session.close()
                except:
                    pass
            
            # Save call session
            if call_session.session_logger:
                call_session.session_logger.log_transcript(
                    "system", "Call ended"
                )
                call_session.session_logger.save_session()
                call_session.session_logger.close()
            
            # Remove from active calls
            del self.active_calls[channel_id]
            
            logger.info(f"Call {channel_id} ended")
    
    async def handle_dtmf(self, event: Dict[str, Any]):
        """Handle DTMF tones (keypad presses)"""
        channel_id = event["channel"]["id"]
        digit = event["digit"]
        
        if channel_id in self.active_calls:
            call_session = self.active_calls[channel_id]
            call_session.session_logger.log_transcript(
                "user_dtmf", f"Pressed: {digit}"
            )
            
            # You can add DTMF handling logic here
            # For example, transfer to different agents based on key press
    
    async def start_voice_session(self, channel_id: str):
        """Start a Google Gemini voice session for the call"""
        if channel_id not in self.active_calls:
            return
            
        call_session = self.active_calls[channel_id]
        
        try:
            # Create a voice agent session connected to the phone call
            voice_session = TelephonyAudioBridge(
                channel_id=channel_id,
                ari_client=self.ari_client,
                call_session=call_session,
                voice_client=self.voice_client,
                model=self.model,
                config=self.voice_config,
                audio_converter=self.audio_converter
            )
            
            call_session.voice_agent_session = voice_session
            
            # Start the audio bridge
            await voice_session.start()
            
        except Exception as e:
            logger.error(f"Error starting voice session for call {channel_id}: {e}")
            # Hang up the call if voice session fails
            self.ari_client.hangup_call(channel_id)
    
    async def originate_call(self, phone_number: str, caller_id: str = None):
        """Make an outbound call through the SIM gateway"""
        # Format for SIP endpoint through SIM gateway
        # This will depend on your Asterisk configuration
        endpoint = f"SIP/{phone_number}"
        
        channel_info = self.ari_client.originate_call(endpoint, caller_id)
        
        if channel_info:
            channel_id = channel_info["id"]
            
            # Create outbound call session
            call_session = CallSession(
                call_id=channel_id,
                caller_id=phone_number,
                call_direction="outbound",
                start_time=datetime.now(timezone.utc)
            )
            
            call_session.session_logger = SessionLogger()
            call_session.session_logger.log_transcript(
                "system", f"Outbound call to {phone_number}"
            )
            
            self.active_calls[channel_id] = call_session
            
            logger.info(f"Initiated outbound call to {phone_number}")
            return channel_id
        
        return None
    
    async def hangup_call(self, channel_id: str):
        """Hang up a specific call"""
        if channel_id in self.active_calls:
            self.ari_client.hangup_call(channel_id)

class TelephonyAudioBridge:
    """Bridges audio between Asterisk channel and Google Gemini voice session"""
    
    def __init__(self, channel_id: str, ari_client: AsteriskARIClient, 
                 call_session: CallSession, voice_client, model: str, 
                 config: dict, audio_converter: AudioConverter):
        self.channel_id = channel_id
        self.ari_client = ari_client
        self.call_session = call_session
        self.voice_client = voice_client
        self.model = model
        self.config = config
        self.audio_converter = audio_converter
        
        self.voice_session = None
        self.audio_queues = {
            "to_gemini": asyncio.Queue(),
            "from_gemini": asyncio.Queue(),
            "to_asterisk": asyncio.Queue()
        }
    
    async def start(self):
        """Start the audio bridge between telephony and voice agent"""
        try:
            # Connect to Google Gemini voice session
            async with self.voice_client.aio.live.connect(
                model=self.model, config=self.config
            ) as session:
                self.voice_session = session
                
                # Start audio processing tasks
                async with asyncio.TaskGroup() as tg:
                    # Task to receive audio from Asterisk and send to Gemini
                    tg.create_task(self.asterisk_to_gemini())
                    
                    # Task to receive audio from Gemini and send to Asterisk  
                    tg.create_task(self.gemini_to_asterisk())
                    
                    # Task to handle Gemini responses
                    tg.create_task(self.handle_gemini_responses())
                    
                    # Placeholder for actual audio I/O with Asterisk
                    # This would need to be implemented based on your specific
                    # Asterisk configuration (ARI external media, etc.)
                    
        except Exception as e:
            logger.error(f"Error in audio bridge for call {self.channel_id}: {e}")
    
    async def asterisk_to_gemini(self):
        """Receive audio from Asterisk channel and send to Gemini"""
        # This is a placeholder - actual implementation depends on 
        # how you're receiving audio from Asterisk (ARI external media, etc.)
        pass
    
    async def gemini_to_asterisk(self):
        """Receive audio from Gemini and send to Asterisk channel"""
        # This is a placeholder - actual implementation depends on
        # how you're sending audio to Asterisk
        pass
    
    async def handle_gemini_responses(self):
        """Handle text responses from Gemini voice session"""
        while True:
            turn = self.voice_session.receive()
            current_response_text = ""
            
            async for response in turn:
                if response.data:
                    # Audio data - convert and queue for Asterisk
                    converted_audio = self.audio_converter.gemini_to_telephony(
                        response.data
                    )
                    await self.audio_queues["to_asterisk"].put(converted_audio)
                    
                if response.text:
                    current_response_text += response.text
            
            # Log complete response
            if current_response_text.strip():
                self.call_session.session_logger.log_transcript(
                    "assistant_response", current_response_text.strip()
                )

# Configuration loader
def load_asterisk_config(config_file: str = "asterisk_config.json") -> AsteriskConfig:
    """Load Asterisk configuration from file"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        # Create default config file
        default_config = {
            "host": "YOUR_SIM_GATEWAY_IP",
            "username": "admin", 
            "password": "YOUR_PASSWORD",
            "sip_port": 5060,
            "ari_port": 8088,
            "ari_username": "asterisk",
            "ari_password": "asterisk",
            "context": "voice-agent"
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default config file: {config_file}")
        logger.info("Please update the configuration with your SIM gateway details")
        
        return AsteriskConfig(**default_config)
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return AsteriskConfig(**config_data)

# CLI interface
async def main():
    """Main function to run the telephony voice agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Telephony Voice Agent")
    parser.add_argument("--config", default="asterisk_config.json", 
                       help="Configuration file path")
    parser.add_argument("--call", help="Make an outbound call to this number")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_asterisk_config(args.config)
    
    # Create and start the telephony voice agent
    agent = TelephonyVoiceAgent(config)
    
    if args.call:
        # Make an outbound call
        logger.info(f"Making outbound call to {args.call}")
        await agent.start()
        await agent.originate_call(args.call)
        
        # Keep running to handle the call
        try:
            await asyncio.sleep(3600)  # Run for 1 hour max
        except KeyboardInterrupt:
            logger.info("Call terminated by user")
    else:
        # Start in server mode to handle incoming calls
        await agent.start()

if __name__ == "__main__":
    asyncio.run(main())