# -*- coding: utf-8 -*-
"""
AGI Voice Server - Simpler alternative to ARI integration
This creates a FastAPI web server that can handle Asterisk AGI requests
and bridge them with the Google Gemini voice agent.
"""

import asyncio
import json
import logging
import uuid
import wave
import io
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from pydub import AudioSegment
import pyaudio

# Import our existing voice agent components
from main import SessionLogger, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, FORMAT, CHANNELS
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent AGI Server", version="1.0.0")

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
    "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}}
}
MODEL = "models/gemini-2.0-flash-live-001"

class AGIVoiceSession:
    """Represents a voice session for an AGI call"""
    
    def __init__(self, session_id: str, caller_id: str, called_number: str):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        self.session_logger = SessionLogger()
        self.voice_session = None
        self.audio_buffer = []
        
        # Log call start
        self.session_logger.log_transcript(
            "system", 
            f"Call started - From: {caller_id}, To: {called_number}"
        )
    
    async def initialize_voice_session(self):
        """Initialize the Google Gemini voice session"""
        try:
            self.voice_session = await voice_client.aio.live.connect(
                model=MODEL, 
                config=VOICE_CONFIG
            ).__aenter__()
            
            # Send initial greeting
            greeting = "Hello! I'm your AI assistant. How can I help you today?"
            await self.voice_session.send(text=greeting)
            
            logger.info(f"Voice session initialized for {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice session: {e}")
            return False
    
    async def process_audio(self, audio_data: bytes) -> bytes:
        """Process incoming audio through the voice agent"""
        try:
            if not self.voice_session:
                await self.initialize_voice_session()
            
            # Convert telephony audio to Gemini format if needed
            processed_audio = self.convert_telephony_to_gemini(audio_data)
            
            # Send audio to Gemini
            await self.voice_session.send(
                input={"data": processed_audio, "mime_type": "audio/pcm"}
            )
            
            # Get response
            response_audio = b""
            turn = self.voice_session.receive()
            async for response in turn:
                if response.data:
                    response_audio += response.data
                if response.text:
                    self.session_logger.log_transcript(
                        "assistant_response", response.text
                    )
            
            # Convert back to telephony format
            return self.convert_gemini_to_telephony(response_audio)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return b""
    
    def convert_telephony_to_gemini(self, audio_data: bytes) -> bytes:
        """Convert 8kHz telephony audio to 16kHz for Gemini"""
        try:
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=8000,  # Typical telephony rate
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

@app.post("/agi/new_call")
async def handle_new_call(call_data: dict, background_tasks: BackgroundTasks):
    """Handle new call from Asterisk AGI"""
    try:
        session_id = str(uuid.uuid4())
        caller_id = call_data.get("caller_id", "Unknown")
        called_number = call_data.get("called_number", "")
        channel = call_data.get("channel", "")
        
        logger.info(f"New call: {caller_id} -> {called_number} on {channel}")
        
        # Create voice session
        voice_session = AGIVoiceSession(session_id, caller_id, called_number)
        active_sessions[session_id] = {
            "voice_session": voice_session,
            "channel": channel,
            "status": "active"
        }
        
        # Initialize voice session in background
        background_tasks.add_task(voice_session.initialize_voice_session)
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Call session created"
        }
        
    except Exception as e:
        logger.error(f"Error handling new call: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agi/audio/{session_id}")
async def process_audio_chunk(session_id: str, audio_data: dict):
    """Process audio chunk from Asterisk"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = active_sessions[session_id]
        voice_session = session_info["voice_session"]
        
        # Decode audio data (assuming base64 encoded)
        audio_bytes = base64.b64decode(audio_data.get("audio", ""))
        
        # Process through voice agent
        response_audio = await voice_session.process_audio(audio_bytes)
        
        # Return processed audio
        return {
            "status": "success",
            "audio": base64.b64encode(response_audio).decode() if response_audio else "",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error processing audio for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agi/dtmf/{session_id}")
async def handle_dtmf(session_id: str, dtmf_data: dict):
    """Handle DTMF (keypad) input"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = active_sessions[session_id]
        voice_session = session_info["voice_session"]
        
        digit = dtmf_data.get("digit", "")
        logger.info(f"DTMF received for session {session_id}: {digit}")
        
        # Log DTMF
        voice_session.session_logger.log_transcript("user_dtmf", f"Pressed: {digit}")
        
        # Handle special DTMF commands
        response = {"status": "success", "action": "continue"}
        
        if digit == "0":
            # Transfer to human operator
            response["action"] = "transfer"
            response["target"] = "operator"
        elif digit == "*":
            # Repeat last message
            response["action"] = "repeat"
        elif digit == "#":
            # End call
            response["action"] = "hangup"
        
        return response
        
    except Exception as e:
        logger.error(f"Error handling DTMF for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agi/end_call/{session_id}")
async def end_call(session_id: str, background_tasks: BackgroundTasks):
    """Handle call termination"""
    try:
        if session_id not in active_sessions:
            return {"status": "success", "message": "Session not found"}
        
        session_info = active_sessions[session_id]
        voice_session = session_info["voice_session"]
        
        logger.info(f"Ending call session {session_id}")
        
        # Cleanup session in background
        background_tasks.add_task(voice_session.cleanup)
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        return {
            "status": "success",
            "message": "Call session ended"
        }
        
    except Exception as e:
        logger.error(f"Error ending call session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/make_call")
async def make_outbound_call(call_request: dict):
    """API endpoint to initiate an outbound call"""
    try:
        phone_number = call_request.get("phone_number")
        caller_id = call_request.get("caller_id", "VoiceAgent")
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="Phone number required")
        
        # This would integrate with your Asterisk system to originate the call
        # For now, return a placeholder response
        session_id = str(uuid.uuid4())
        
        return {
            "status": "success",
            "session_id": session_id,
            "phone_number": phone_number,
            "message": "Outbound call initiated"
        }
        
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
                "channel": session_info["channel"],
                "status": session_info["status"]
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
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.websocket("/ws/audio/{session_id}")
async def websocket_audio(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()
    
    try:
        if session_id not in active_sessions:
            await websocket.send_json({"error": "Session not found"})
            return
        
        session_info = active_sessions[session_id]
        voice_session = session_info["voice_session"]
        
        while True:
            # Receive audio data via WebSocket
            data = await websocket.receive_bytes()
            
            # Process through voice agent
            response_audio = await voice_session.process_audio(data)
            
            # Send response back
            if response_audio:
                await websocket.send_bytes(response_audio)
                
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI Voice Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting AGI Voice Server on {args.host}:{args.port}")
    logger.info("Endpoints available:")
    logger.info("  POST /agi/new_call - Handle new call from Asterisk")
    logger.info("  POST /agi/audio/{session_id} - Process audio chunk")
    logger.info("  POST /agi/dtmf/{session_id} - Handle DTMF input")
    logger.info("  POST /agi/end_call/{session_id} - End call session")
    logger.info("  POST /api/make_call - Initiate outbound call")
    logger.info("  GET /api/sessions - Get active sessions")
    logger.info("  GET /health - Health check")
    logger.info("  WS /ws/audio/{session_id} - Real-time audio WebSocket")
    
    uvicorn.run(
        "agi_voice_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )