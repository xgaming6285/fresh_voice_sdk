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
import warnings
# Suppress audioop deprecation warning
warnings.filterwarnings('ignore', message='.*audioop.*', category=DeprecationWarning)
import audioop
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import queue
# import scipy.signal
# from scipy.signal import resample_poly, butter, sosfilt, sosfilt_zi

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Configure logging with millisecond-precision timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress repetitive Google Generative AI warnings about inline_data (we intentionally use audio)
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

import pyaudio
import requests
# import numpy as np
import webrtcvad  # High-performance VAD

# Asterisk AMI Monitor for linkedid capture
from asterisk_ami_monitor import start_ami_monitoring, get_current_linkedid, ami_monitor

# Multi-tier transcription system with Windows-compatible fallbacks
TRANSCRIPTION_METHOD = None
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False
OPENAI_API_AVAILABLE = False
GEMINI_API_AVAILABLE = False

# Check for Gemini API (preferred - fast and accurate)
try:
    from google import genai
    # Gemini API is always available if google-genai is installed
    GEMINI_API_AVAILABLE = True
    TRANSCRIPTION_METHOD = "gemini_api"
    logger.info("✅ Gemini API available for transcription (fast cloud-based transcription)")
    logger.info("🚀 Using Gemini for transcription - much faster than Whisper large model")
except ImportError:
    logger.info("💡 Gemini API not available, checking other methods...")
except Exception as e:
    logger.warning(f"⚠️ Gemini API setup failed: {e}")

# Try faster-whisper (Windows-friendly, no numba dependency) - only if Gemini not available
if not GEMINI_API_AVAILABLE:
    try:
        from faster_whisper import WhisperModel
        FASTER_WHISPER_AVAILABLE = True
        TRANSCRIPTION_METHOD = "faster_whisper"
        logger.info("✅ faster-whisper library loaded successfully (recommended for Windows)")
    except ImportError:
        logger.info("💡 faster-whisper not available, trying openai-whisper...")
    except Exception as e:
        logger.warning(f"⚠️ faster-whisper failed to load: {e}")

# Try original openai-whisper (local, better for Bulgarian)
if not GEMINI_API_AVAILABLE and not FASTER_WHISPER_AVAILABLE:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        TRANSCRIPTION_METHOD = "openai_whisper"
        logger.info("✅ openai-whisper library loaded for local transcription")
        logger.info("🎯 Using LOCAL Whisper model (no API quotas or internet required)")
    except ImportError as e:
        logger.warning(f"⚠️ openai-whisper library not available: {e}")
    except OSError as e:
        logger.warning(f"⚠️ openai-whisper failed to load (Windows compatibility issue): {e}")
        logger.warning("📝 This is often due to numba/llvmlite compatibility on Windows")
    except Exception as e:
        logger.warning(f"⚠️ Unexpected error loading openai-whisper: {e}")

# Fallback to OpenAI API only if nothing else is available
if not GEMINI_API_AVAILABLE and not FASTER_WHISPER_AVAILABLE and not WHISPER_AVAILABLE:
    try:
        import openai
        # Check if API key is available in environment or can be configured
        import os
        if os.getenv('OPENAI_API_KEY'):
            OPENAI_API_AVAILABLE = True
            TRANSCRIPTION_METHOD = "openai_api"
            logger.info("✅ OpenAI API available for transcription (cloud-based fallback)")
        else:
            logger.info("💡 OpenAI API client available but no API key configured")
    except ImportError:
        logger.info("💡 OpenAI API client not available")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI API setup failed: {e}")

# Final status check
TRANSCRIPTION_AVAILABLE = GEMINI_API_AVAILABLE or FASTER_WHISPER_AVAILABLE or OPENAI_API_AVAILABLE or WHISPER_AVAILABLE
if TRANSCRIPTION_AVAILABLE:
    logger.info(f"🎤 Transcription enabled using: {TRANSCRIPTION_METHOD}")
else:
    logger.warning("⚠️ No transcription method available - transcription features will be disabled")
    logger.warning("💡 To enable transcription, install: pip install google-genai")
    logger.warning("💡 Or for local transcription: pip install faster-whisper")


# =============================================================================
# PROFESSIONAL COLD CALLING ENHANCEMENTS
# =============================================================================

from enum import Enum, auto

class ConversationState(Enum):
    """
    Explicit conversation state machine for professional turn-taking.
    
    Professional cold calling systems track conversation state explicitly
    to prevent issues like:
    - Sending end-of-turn while AI is speaking
    - Interrupting the greeting
    - Missing user responses due to noise
    """
    INITIALIZING = auto()      # Session starting up
    GREETING = auto()          # AI speaking first (greeting)
    LISTENING = auto()         # Waiting for user response
    USER_SPEAKING = auto()     # User is actively talking
    PROCESSING = auto()        # Detecting end-of-turn, brief pause
    AI_RESPONDING = auto()     # AI generating/speaking response
    INTERRUPTED = auto()       # User interrupted AI
    CALL_ENDING = auto()       # Goodbye detected, wrapping up
    ENDED = auto()             # Call terminated


class AnsweringMachineDetector:
    """
    Detects answering machines/voicemail within the first few seconds.
    
    Professional cold calling systems detect voicemail to:
    1. Avoid wasting agent time on voicemail
    2. Leave appropriate voicemail messages
    3. Schedule callback for live calls
    
    Detection indicators:
    - Long continuous speech (>5s without pause) - voicemail greeting
    - Specific phrases: "leave a message", "beep", "not available"
    - Monotone energy pattern (recorded vs live)
    - No interruption response (live humans respond to pauses)
    """
    
    # Phrases indicating answering machine (multi-language)
    AMD_PHRASES = re.compile(r"""
        (
            # English
            leave\s+(a\s+)?message|
            after\s+the\s+(beep|tone)|
            not\s+available|
            voicemail|
            mailbox|
            record\s+your\s+message|
            press\s+\d+\s+to|
            
            # Bulgarian
            оставете\s+съобщение|
            след\s+сигнала|
            гласова\s+поща|
            не\s+е\s+на\s+разположение|
            натиснете|
            
            # Spanish  
            deje\s+su\s+mensaje|
            después\s+del\s+tono|
            buzón\s+de\s+voz|
            
            # German
            hinterlassen\s+sie|
            nach\s+dem\s+signalton|
            mailbox|
            
            # French
            laissez\s+un\s+message|
            après\s+le\s+bip|
            messagerie
        )
    """, re.I | re.VERBOSE)
    
    def __init__(self, detection_window_sec: float = 5.0):
        """
        Args:
            detection_window_sec: Time window for AMD detection (default 5s)
        """
        self.detection_window_sec = detection_window_sec
        self.call_start_time = None
        self.continuous_speech_start = None
        self.speech_segments = []  # List of (start, end) tuples
        self.detected_result = None  # None = undetermined, True = AMD, False = Human
        self.transcript_buffer = ""
        self.energy_samples = []  # For monotone detection
        self.last_pause_time = None
        self.pause_count = 0
        
        # Thresholds
        self.continuous_speech_threshold = 4.5  # Seconds of continuous speech = AMD
        self.min_pauses_for_human = 2  # Humans typically pause at least twice in 5s
        self.energy_variance_threshold = 500  # Low variance = recorded voice
        
        logger.info(f"📞 AMD initialized (window={detection_window_sec}s, continuous_threshold={self.continuous_speech_threshold}s)")
    
    def start_detection(self):
        """Call when call is answered to start AMD detection window"""
        self.call_start_time = time.time()
        self.continuous_speech_start = None
        self.speech_segments = []
        self.detected_result = None
        self.transcript_buffer = ""
        self.energy_samples = []
        self.last_pause_time = time.time()
        self.pause_count = 0
        logger.info("📞 AMD detection started")
    
    def feed_audio(self, is_speech: bool, energy: float) -> Optional[bool]:
        """
        Feed audio analysis results to AMD detector.
        
        Args:
            is_speech: Whether current frame contains speech
            energy: RMS energy of current frame
            
        Returns:
            None if still detecting, True if AMD detected, False if human detected
        """
        if self.call_start_time is None:
            return None
            
        if self.detected_result is not None:
            return self.detected_result
            
        current_time = time.time()
        elapsed = current_time - self.call_start_time
        
        # Past detection window - default to human
        if elapsed > self.detection_window_sec:
            if self.detected_result is None:
                self.detected_result = False
                logger.info(f"📞 AMD: Detection window ended - assuming HUMAN (no clear AMD indicators)")
            return self.detected_result
        
        # Track energy for variance analysis
        self.energy_samples.append(energy)
        
        if is_speech:
            if self.continuous_speech_start is None:
                self.continuous_speech_start = current_time
            
            # Check for continuous speech (AMD indicator)
            speech_duration = current_time - self.continuous_speech_start
            if speech_duration >= self.continuous_speech_threshold:
                self.detected_result = True
                logger.info(f"📞 AMD DETECTED: Continuous speech for {speech_duration:.1f}s (threshold: {self.continuous_speech_threshold}s)")
                return True
        else:
            # Silence/pause detected
            if self.continuous_speech_start is not None:
                # End of speech segment
                segment_duration = current_time - self.continuous_speech_start
                if segment_duration > 0.3:  # Only count significant speech segments
                    self.speech_segments.append((self.continuous_speech_start, current_time))
                self.continuous_speech_start = None
                
                # Count pause
                if current_time - self.last_pause_time > 0.5:  # Significant pause
                    self.pause_count += 1
                    self.last_pause_time = current_time
                    
                    # Multiple pauses suggest human (conversational pattern)
                    if self.pause_count >= self.min_pauses_for_human and elapsed > 2.0:
                        self.detected_result = False
                        logger.info(f"📞 AMD: HUMAN detected (conversational pauses: {self.pause_count})")
                        return False
        
        return None
    
    def feed_transcript(self, text: str) -> Optional[bool]:
        """
        Feed transcript text for phrase-based AMD detection.
        
        Args:
            text: Transcribed text from the call
            
        Returns:
            True if AMD phrases detected, None otherwise
        """
        if self.detected_result is not None:
            return self.detected_result
            
        self.transcript_buffer += " " + text.lower()
        
        # Check for AMD phrases
        if self.AMD_PHRASES.search(self.transcript_buffer):
            self.detected_result = True
            logger.info(f"📞 AMD DETECTED: Voicemail phrase detected in transcript")
            return True
            
        return None
    
    def get_result(self) -> Optional[bool]:
        """Get current detection result"""
        return self.detected_result
    
    def is_detecting(self) -> bool:
        """Check if still in detection window"""
        if self.call_start_time is None:
            return False
        return (time.time() - self.call_start_time) < self.detection_window_sec and self.detected_result is None


class AdaptiveSilenceDetector:
    """
    Professional silence detection with adaptive thresholds.
    
    Instead of fixed thresholds, this adapts to:
    - Ambient noise level of the call
    - GSM/telephony line quality
    - Speaking patterns of the current caller
    """
    
    def __init__(self, 
                 initial_threshold: float = 5000.0,
                 min_threshold: float = 2000.0,
                 max_threshold: float = 12000.0,
                 adaptation_rate: float = 0.05):
        """
        Args:
            initial_threshold: Starting energy threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            adaptation_rate: How fast to adapt (0.01-0.1)
        """
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate
        
        # Running statistics
        self.noise_floor = initial_threshold * 0.3  # Estimated background noise
        self.speech_level = initial_threshold * 2.0  # Estimated speech level
        self.energy_history = []
        self.max_history = 100  # Keep last 100 samples
        
        # Calibration state
        self.calibrated = False
        self.calibration_samples = 0
        self.calibration_target = 50  # Samples needed for initial calibration
        
        logger.info(f"📊 Adaptive silence detector initialized (threshold={initial_threshold})")
    
    def update(self, energy: float, is_speech: bool) -> float:
        """
        Update adaptive threshold based on observed energy.
        
        Args:
            energy: RMS energy of current frame
            is_speech: Whether WebRTC VAD detected speech
            
        Returns:
            Current adaptive threshold
        """
        # Add to history
        self.energy_history.append((energy, is_speech))
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
        
        # Calibration phase - gather statistics
        if not self.calibrated:
            self.calibration_samples += 1
            if self.calibration_samples >= self.calibration_target:
                self._calibrate()
            return self.threshold
        
        # Adaptive update
        if is_speech:
            # Update speech level estimate (slow adaptation)
            self.speech_level = (1 - self.adaptation_rate * 0.5) * self.speech_level + \
                               (self.adaptation_rate * 0.5) * energy
        else:
            # Update noise floor estimate (faster adaptation for noise)
            self.noise_floor = (1 - self.adaptation_rate) * self.noise_floor + \
                              self.adaptation_rate * energy
        
        # Set threshold between noise floor and speech level
        # Threshold = noise_floor + 40% of (speech_level - noise_floor)
        new_threshold = self.noise_floor + 0.4 * (self.speech_level - self.noise_floor)
        
        # Clamp to bounds
        self.threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        return self.threshold
    
    def _calibrate(self):
        """Perform initial calibration based on collected samples"""
        if not self.energy_history:
            self.calibrated = True
            return
            
        speech_energies = [e for e, is_speech in self.energy_history if is_speech]
        noise_energies = [e for e, is_speech in self.energy_history if not is_speech]
        
        if speech_energies:
            self.speech_level = sum(speech_energies) / len(speech_energies)
        if noise_energies:
            self.noise_floor = sum(noise_energies) / len(noise_energies)
        
        # Set initial threshold
        self.threshold = self.noise_floor + 0.4 * (self.speech_level - self.noise_floor)
        self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))
        
        self.calibrated = True
        logger.info(f"📊 Adaptive threshold calibrated: noise={self.noise_floor:.0f}, speech={self.speech_level:.0f}, threshold={self.threshold:.0f}")
    
    def get_threshold(self) -> float:
        """Get current adaptive threshold"""
        return self.threshold
    
    def is_silence(self, energy: float) -> bool:
        """Check if energy level indicates silence"""
        return energy < self.threshold


class DebouncedEndOfTurnDetector:
    """
    Professional end-of-turn detection with debouncing.
    
    Instead of triggering on first silence, requires:
    1. Minimum silence duration (400ms default)
    2. Multiple confirmation frames
    3. State-aware triggering (don't trigger while AI speaking)
    """
    
    def __init__(self,
                 silence_threshold_sec: float = 0.40,  # 400ms - professional standard
                 confirmation_frames: int = 3,         # Require 3 consecutive silence frames
                 cooldown_sec: float = 0.5):           # Cooldown after triggering
        """
        Args:
            silence_threshold_sec: Required silence duration to trigger end-of-turn
            confirmation_frames: Number of consecutive silence frames required
            cooldown_sec: Minimum time between end-of-turn triggers
        """
        self.silence_threshold_sec = silence_threshold_sec
        self.confirmation_frames = confirmation_frames
        self.cooldown_sec = cooldown_sec
        
        # State tracking
        self.silence_start_time = None
        self.consecutive_silence_frames = 0
        self.last_trigger_time = 0
        self.triggered = False
        self.speech_detected_since_trigger = False
        
        logger.info(f"⏱️ Debounced EOT detector: silence={silence_threshold_sec}s, confirm={confirmation_frames} frames, cooldown={cooldown_sec}s")
    
    def update(self, is_speech: bool, conversation_state: ConversationState) -> bool:
        """
        Update detector with new audio frame analysis.
        
        Args:
            is_speech: Whether current frame contains speech
            conversation_state: Current conversation state
            
        Returns:
            True if end-of-turn should be triggered
        """
        current_time = time.time()
        
        # Don't detect end-of-turn in these states
        if conversation_state in (ConversationState.GREETING, 
                                   ConversationState.AI_RESPONDING,
                                   ConversationState.INITIALIZING,
                                   ConversationState.CALL_ENDING,
                                   ConversationState.ENDED):
            self._reset()
            return False
        
        # Check cooldown
        if current_time - self.last_trigger_time < self.cooldown_sec:
            return False
        
        if is_speech:
            # Speech detected - reset silence tracking
            self.silence_start_time = None
            self.consecutive_silence_frames = 0
            self.triggered = False
            self.speech_detected_since_trigger = True
            return False
        
        # Silence detected
        if self.silence_start_time is None:
            self.silence_start_time = current_time
            self.consecutive_silence_frames = 1
        else:
            self.consecutive_silence_frames += 1
        
        # Check if we should trigger
        silence_duration = current_time - self.silence_start_time
        
        if (silence_duration >= self.silence_threshold_sec and 
            self.consecutive_silence_frames >= self.confirmation_frames and
            self.speech_detected_since_trigger and
            not self.triggered):
            
            self.triggered = True
            self.last_trigger_time = current_time
            self.speech_detected_since_trigger = False
            
            logger.debug(f"⏱️ EOT triggered: silence={silence_duration:.2f}s, frames={self.consecutive_silence_frames}")
            return True
        
        return False
    
    def _reset(self):
        """Reset detector state"""
        self.silence_start_time = None
        self.consecutive_silence_frames = 0
        self.triggered = False
    
    def reset_for_new_turn(self):
        """Reset for a new conversation turn"""
        self._reset()
        self.speech_detected_since_trigger = False


class CallQualityTracker:
    """
    Professional call quality and metrics tracking.
    
    Tracks key performance indicators for cold calling optimization:
    - Response latency (time from user stop speaking to AI response)
    - Turn-taking efficiency (interruptions, overlap)
    - Conversation flow (back-and-forth ratio)
    - Call outcomes for analytics
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.call_start_time = time.time()
        
        # Latency tracking
        self.response_latencies = []  # Time from user EOT to AI first audio
        self.user_eot_time = None
        
        # Turn-taking metrics
        self.user_turns = 0
        self.ai_turns = 0
        self.interruptions_by_user = 0
        self.interruptions_by_ai = 0
        self.overlapping_speech_events = 0
        
        # Speech metrics
        self.total_user_speech_time = 0.0
        self.total_ai_speech_time = 0.0
        self.longest_user_utterance = 0.0
        self.longest_ai_utterance = 0.0
        
        # Outcome tracking
        self.amd_result = None  # True=AMD, False=Human, None=Unknown
        self.call_outcome = None  # "completed", "voicemail", "no_answer", "hung_up", etc.
        self.goodbye_detected = False
        
        logger.info(f"📊 Call quality tracker initialized for {session_id}")
    
    def on_user_end_of_turn(self):
        """Called when user stops speaking"""
        self.user_eot_time = time.time()
        self.user_turns += 1
    
    def on_ai_start_response(self):
        """Called when AI starts responding"""
        if self.user_eot_time:
            latency = time.time() - self.user_eot_time
            self.response_latencies.append(latency)
            self.user_eot_time = None
            
            # Log exceptional latencies
            if latency > 2.0:
                logger.warning(f"📊 High response latency: {latency:.2f}s")
        
        self.ai_turns += 1
    
    def on_user_interrupt(self):
        """Called when user interrupts AI"""
        self.interruptions_by_user += 1
    
    def on_speech_overlap(self):
        """Called when user and AI are both speaking"""
        self.overlapping_speech_events += 1
    
    def set_amd_result(self, is_amd: bool):
        """Set answering machine detection result"""
        self.amd_result = is_amd
    
    def set_call_outcome(self, outcome: str):
        """Set final call outcome"""
        self.call_outcome = outcome
    
    def get_metrics(self) -> dict:
        """Get all call quality metrics"""
        call_duration = time.time() - self.call_start_time
        avg_latency = sum(self.response_latencies) / len(self.response_latencies) if self.response_latencies else 0
        max_latency = max(self.response_latencies) if self.response_latencies else 0
        
        return {
            "session_id": self.session_id,
            "call_duration_sec": round(call_duration, 2),
            "avg_response_latency_sec": round(avg_latency, 3),
            "max_response_latency_sec": round(max_latency, 3),
            "user_turns": self.user_turns,
            "ai_turns": self.ai_turns,
            "turn_ratio": round(self.user_turns / max(1, self.ai_turns), 2),
            "interruptions_by_user": self.interruptions_by_user,
            "overlap_events": self.overlapping_speech_events,
            "amd_result": "AMD" if self.amd_result else ("Human" if self.amd_result is False else "Unknown"),
            "call_outcome": self.call_outcome or "in_progress"
        }
    
    def log_summary(self):
        """Log a summary of call quality metrics"""
        metrics = self.get_metrics()
        logger.info(f"📊 CALL QUALITY SUMMARY for {self.session_id}:")
        logger.info(f"   Duration: {metrics['call_duration_sec']}s | Turns: User={metrics['user_turns']}, AI={metrics['ai_turns']}")
        logger.info(f"   Latency: Avg={metrics['avg_response_latency_sec']}s, Max={metrics['max_response_latency_sec']}s")
        logger.info(f"   Interrupts: {metrics['interruptions_by_user']} | AMD: {metrics['amd_result']} | Outcome: {metrics['call_outcome']}")


# =============================================================================
# LATENCY TRACKER - Comprehensive Pipeline Latency Monitoring
# =============================================================================

class LatencyTracker:
    """
    Comprehensive latency tracking for the voice pipeline.
    
    Tracks timing at each stage:
    - RTP Receive: Audio packet received from SIM gate (Bulgaria)
    - Decode: After μ-law/A-law decoding
    - Preprocess: After audio preprocessing pipeline (bandpass, AGC, etc.)
    - VAD: After Voice Activity Detection filtering
    - Gemini Send: When audio is sent to Gemini API
    - Gemini First Response: First audio byte received from Gemini
    - Convert: After converting response to telephony format
    - RTP Send: When response is sent back to SIM gate
    
    Network topology:
    - SIM Gate (Bulgaria) <-> UK VPS (Agent) <-> Google Gemini API
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.call_start_time = time.time()
        
        # === INBOUND LATENCY (SIM Gate -> Gemini) ===
        self.rtp_receive_times = []           # When RTP packet received
        self.decode_times = []                # After μ-law decoding
        self.preprocess_times = []            # After audio preprocessing
        self.vad_times = []                   # After VAD filtering
        self.gemini_send_times = []           # When sent to Gemini
        
        # === OUTBOUND LATENCY (Gemini -> SIM Gate) ===
        self.gemini_response_times = []       # First byte from Gemini
        self.convert_times = []               # After telephony conversion
        self.rtp_send_times = []              # When sent via RTP
        
        # === ROUND-TRIP TRACKING ===
        # Maps utterance_id -> {stage: timestamp}
        self.utterance_tracking = {}
        self.current_utterance_id = 0
        self.last_speech_start_time = None
        self.last_user_eot_time = None        # End of turn marker
        
        # === STAGE LATENCIES (computed deltas) ===
        self.stage_latencies = {
            'rtp_to_decode': [],              # Network receive -> decode
            'decode_to_preprocess': [],       # Decode -> preprocessing
            'preprocess_to_vad': [],          # Preprocessing -> VAD
            'vad_to_gemini_send': [],         # VAD -> Gemini send
            'gemini_roundtrip': [],           # Send -> first response
            'gemini_to_convert': [],          # Response -> conversion
            'convert_to_rtp_send': [],        # Conversion -> RTP send
            'total_inbound': [],              # RTP receive -> Gemini send
            'total_outbound': [],             # Gemini response -> RTP send
            'total_roundtrip': [],            # User speech -> AI audio out
            'user_eot_to_ai_audio': [],       # User end-of-turn -> first AI audio
        }
        
        # === REPORTING STATE ===
        self.last_report_time = time.time()
        self.report_interval_sec = 10.0       # Log summary every 10 seconds
        self.packets_tracked = 0
        self.responses_tracked = 0
        
        # === NETWORK LATENCY ESTIMATES ===
        self.estimated_network_latency_ms = {
            'sim_gate_to_agent': None,        # Bulgaria -> UK
            'agent_to_gemini': None,          # UK -> Gemini API
        }
        
        logger.info(f"⏱️ Latency tracker initialized for session {session_id}")
    
    def start_utterance(self) -> int:
        """Start tracking a new user utterance. Returns utterance_id."""
        self.current_utterance_id += 1
        self.last_speech_start_time = time.time()
        self.utterance_tracking[self.current_utterance_id] = {
            'speech_start': self.last_speech_start_time,
        }
        return self.current_utterance_id
    
    def mark_rtp_receive(self, packet_size: int = 0):
        """Called when RTP packet is received from SIM gate."""
        t = time.time()
        self.rtp_receive_times.append(t)
        self.packets_tracked += 1
        
        if self.current_utterance_id in self.utterance_tracking:
            if 'rtp_receive' not in self.utterance_tracking[self.current_utterance_id]:
                self.utterance_tracking[self.current_utterance_id]['rtp_receive'] = t
    
    def mark_decode_complete(self):
        """Called after μ-law/A-law decoding."""
        t = time.time()
        self.decode_times.append(t)
        
        # Calculate delta from RTP receive
        if self.rtp_receive_times:
            delta = (t - self.rtp_receive_times[-1]) * 1000  # ms
            self.stage_latencies['rtp_to_decode'].append(delta)
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['decode'] = t
    
    def mark_preprocess_complete(self):
        """Called after audio preprocessing (bandpass, AGC, etc.)."""
        t = time.time()
        self.preprocess_times.append(t)
        
        # Calculate delta from decode
        if self.decode_times:
            delta = (t - self.decode_times[-1]) * 1000  # ms
            self.stage_latencies['decode_to_preprocess'].append(delta)
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['preprocess'] = t
    
    def mark_vad_complete(self, is_speech: bool):
        """Called after VAD filtering."""
        t = time.time()
        self.vad_times.append(t)
        
        # Calculate delta from preprocess
        if self.preprocess_times:
            delta = (t - self.preprocess_times[-1]) * 1000  # ms
            self.stage_latencies['preprocess_to_vad'].append(delta)
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['vad'] = t
            self.utterance_tracking[self.current_utterance_id]['is_speech'] = is_speech
    
    def mark_gemini_send(self, chunk_size: int = 0):
        """Called when audio is sent to Gemini."""
        t = time.time()
        self.gemini_send_times.append(t)
        
        # Calculate delta from VAD
        if self.vad_times:
            delta = (t - self.vad_times[-1]) * 1000  # ms
            self.stage_latencies['vad_to_gemini_send'].append(delta)
        
        # Calculate total inbound latency
        if self.rtp_receive_times:
            total = (t - self.rtp_receive_times[-1]) * 1000  # ms
            self.stage_latencies['total_inbound'].append(total)
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['gemini_send'] = t
            self.utterance_tracking[self.current_utterance_id]['chunk_size'] = chunk_size
    
    def mark_user_end_of_turn(self):
        """Called when user stops speaking (end of turn detected)."""
        self.last_user_eot_time = time.time()
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['user_eot'] = self.last_user_eot_time
    
    def mark_gemini_first_response(self, response_size: int = 0):
        """Called when first audio byte is received from Gemini."""
        t = time.time()
        self.gemini_response_times.append(t)
        self.responses_tracked += 1
        
        # Calculate Gemini roundtrip (from last send)
        if self.gemini_send_times:
            delta = (t - self.gemini_send_times[-1]) * 1000  # ms
            self.stage_latencies['gemini_roundtrip'].append(delta)
            
            # Estimate network latency to Gemini (rough: half of roundtrip minus processing)
            # Gemini processing is ~100-300ms, so network is roughly (delta - 200) / 2
            if delta > 200:
                self.estimated_network_latency_ms['agent_to_gemini'] = (delta - 200) / 2
        
        # Calculate user EOT to first AI audio
        if self.last_user_eot_time:
            eot_delta = (t - self.last_user_eot_time) * 1000  # ms
            self.stage_latencies['user_eot_to_ai_audio'].append(eot_delta)
            logger.info(f"⏱️ EOT→AI: {eot_delta:.0f}ms | Gemini roundtrip: {self.stage_latencies['gemini_roundtrip'][-1]:.0f}ms")
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['gemini_response'] = t
    
    def mark_convert_complete(self):
        """Called after converting Gemini audio to telephony format."""
        t = time.time()
        self.convert_times.append(t)
        
        # Calculate delta from Gemini response
        if self.gemini_response_times:
            delta = (t - self.gemini_response_times[-1]) * 1000  # ms
            self.stage_latencies['gemini_to_convert'].append(delta)
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['convert'] = t
    
    def mark_rtp_send(self, packet_size: int = 0):
        """Called when response audio is sent via RTP to SIM gate."""
        t = time.time()
        self.rtp_send_times.append(t)
        
        # Calculate delta from convert
        if self.convert_times:
            delta = (t - self.convert_times[-1]) * 1000  # ms
            self.stage_latencies['convert_to_rtp_send'].append(delta)
        
        # Calculate total outbound latency
        if self.gemini_response_times:
            total = (t - self.gemini_response_times[-1]) * 1000  # ms
            self.stage_latencies['total_outbound'].append(total)
        
        # Calculate total roundtrip (from speech start if available)
        if self.last_speech_start_time:
            roundtrip = (t - self.last_speech_start_time) * 1000  # ms
            self.stage_latencies['total_roundtrip'].append(roundtrip)
        
        if self.current_utterance_id in self.utterance_tracking:
            self.utterance_tracking[self.current_utterance_id]['rtp_send'] = t
        
        # Periodic reporting
        self._maybe_report()
    
    def _calculate_stats(self, values: list) -> dict:
        """Calculate min, max, avg, p50, p95 for a list of values."""
        if not values:
            return {'min': 0, 'max': 0, 'avg': 0, 'p50': 0, 'p95': 0, 'count': 0}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            'min': round(sorted_vals[0], 2),
            'max': round(sorted_vals[-1], 2),
            'avg': round(sum(sorted_vals) / n, 2),
            'p50': round(sorted_vals[n // 2], 2),
            'p95': round(sorted_vals[int(n * 0.95)] if n > 1 else sorted_vals[-1], 2),
            'count': n
        }
    
    def _maybe_report(self):
        """Log latency report if interval has passed."""
        now = time.time()
        if (now - self.last_report_time) >= self.report_interval_sec:
            self.log_latency_report()
            self.last_report_time = now
    
    def get_latency_summary(self) -> dict:
        """Get comprehensive latency statistics."""
        summary = {
            'session_id': self.session_id,
            'duration_sec': round(time.time() - self.call_start_time, 2),
            'packets_tracked': self.packets_tracked,
            'responses_tracked': self.responses_tracked,
            'stages': {},
            'network_estimates_ms': self.estimated_network_latency_ms,
        }
        
        # Calculate stats for each stage
        for stage, values in self.stage_latencies.items():
            summary['stages'][stage] = self._calculate_stats(values)
        
        return summary
    
    def log_latency_report(self):
        """Log a detailed latency report."""
        summary = self.get_latency_summary()
        
        logger.info(f"")
        logger.info(f"╔══════════════════════════════════════════════════════════════════╗")
        logger.info(f"║  ⏱️  LATENCY REPORT - Session {self.session_id[:8]}...              ║")
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Duration: {summary['duration_sec']}s | Packets: {summary['packets_tracked']} | Responses: {summary['responses_tracked']}")
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        logger.info(f"║  INBOUND (SIM Gate → Agent → Gemini):                            ║")
        
        inbound = summary['stages'].get('total_inbound', {})
        if inbound.get('count', 0) > 0:
            logger.info(f"║    Total:      Avg={inbound['avg']:.1f}ms  P95={inbound['p95']:.1f}ms  Max={inbound['max']:.1f}ms")
        
        for stage in ['rtp_to_decode', 'decode_to_preprocess', 'preprocess_to_vad', 'vad_to_gemini_send']:
            stats = summary['stages'].get(stage, {})
            if stats.get('count', 0) > 0:
                stage_name = stage.replace('_to_', '→').replace('_', ' ')
                logger.info(f"║    {stage_name:20}: Avg={stats['avg']:.1f}ms  Max={stats['max']:.1f}ms")
        
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        logger.info(f"║  GEMINI API:                                                      ║")
        
        gemini = summary['stages'].get('gemini_roundtrip', {})
        if gemini.get('count', 0) > 0:
            logger.info(f"║    Roundtrip:  Avg={gemini['avg']:.0f}ms  P95={gemini['p95']:.0f}ms  Max={gemini['max']:.0f}ms")
        
        eot = summary['stages'].get('user_eot_to_ai_audio', {})
        if eot.get('count', 0) > 0:
            logger.info(f"║    EOT→Audio:  Avg={eot['avg']:.0f}ms  P95={eot['p95']:.0f}ms  Max={eot['max']:.0f}ms")
        
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        logger.info(f"║  OUTBOUND (Gemini → Agent → SIM Gate):                           ║")
        
        outbound = summary['stages'].get('total_outbound', {})
        if outbound.get('count', 0) > 0:
            logger.info(f"║    Total:      Avg={outbound['avg']:.1f}ms  P95={outbound['p95']:.1f}ms  Max={outbound['max']:.1f}ms")
        
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        logger.info(f"║  END-TO-END:                                                      ║")
        
        roundtrip = summary['stages'].get('total_roundtrip', {})
        if roundtrip.get('count', 0) > 0:
            logger.info(f"║    Speech→Out: Avg={roundtrip['avg']:.0f}ms  P95={roundtrip['p95']:.0f}ms  Max={roundtrip['max']:.0f}ms")
        
        logger.info(f"╚══════════════════════════════════════════════════════════════════╝")
        logger.info(f"")
    
    def log_final_summary(self):
        """Log final latency summary at call end with bottleneck analysis."""
        summary = self.get_latency_summary()
        
        logger.info(f"")
        logger.info(f"╔══════════════════════════════════════════════════════════════════╗")
        logger.info(f"║  ⏱️  FINAL LATENCY SUMMARY - Session {self.session_id[:8]}...       ║")
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        
        # Identify bottleneck
        bottleneck = None
        max_avg = 0
        for stage, stats in summary['stages'].items():
            if stats.get('avg', 0) > max_avg and 'total' not in stage:
                max_avg = stats['avg']
                bottleneck = stage
        
        if bottleneck:
            logger.info(f"║  🎯 BOTTLENECK IDENTIFIED: {bottleneck.replace('_', ' ').upper()}")
            logger.info(f"║     Average latency: {max_avg:.0f}ms")
            
            # Provide recommendations based on bottleneck
            if 'gemini' in bottleneck.lower():
                logger.info(f"║     → This is Gemini API latency (expected 200-500ms)")
                logger.info(f"║     → Consider: Closer region, model optimization")
            elif 'preprocess' in bottleneck.lower():
                logger.info(f"║     → Audio preprocessing is taking too long")
                logger.info(f"║     → Consider: Reduce FFT size, optimize filters")
            elif 'vad' in bottleneck.lower():
                logger.info(f"║     → VAD processing is slow")
                logger.info(f"║     → Consider: Reduce min_speech_frames")
            elif 'rtp' in bottleneck.lower():
                logger.info(f"║     → Network latency to SIM gate (Bulgaria)")
                logger.info(f"║     → Consider: Closer server, VPN optimization")
        
        logger.info(f"╠══════════════════════════════════════════════════════════════════╣")
        
        # Key metrics summary
        eot = summary['stages'].get('user_eot_to_ai_audio', {})
        gemini = summary['stages'].get('gemini_roundtrip', {})
        inbound = summary['stages'].get('total_inbound', {})
        outbound = summary['stages'].get('total_outbound', {})
        
        logger.info(f"║  KEY METRICS:")
        if eot.get('count', 0) > 0:
            logger.info(f"║    User speaks → AI responds: Avg {eot['avg']:.0f}ms, P95 {eot['p95']:.0f}ms")
        if gemini.get('count', 0) > 0:
            logger.info(f"║    Gemini API roundtrip:      Avg {gemini['avg']:.0f}ms, P95 {gemini['p95']:.0f}ms")
        if inbound.get('count', 0) > 0:
            logger.info(f"║    SIM→Agent→Gemini:         Avg {inbound['avg']:.1f}ms")
        if outbound.get('count', 0) > 0:
            logger.info(f"║    Gemini→Agent→SIM:         Avg {outbound['avg']:.1f}ms")
        
        logger.info(f"╚══════════════════════════════════════════════════════════════════╝")
        
        return summary


class VoiceActivityDetector:
    """
    Enhanced Client-Side VAD using WebRTC VAD with energy tracking.
    
    Professional cold calling improvements:
    1. Filter out background noise/static so Gemini doesn't get interrupted falsely.
    2. Only send actual human speech to Gemini.
    3. Track energy levels for adaptive threshold systems.
    4. Provide detailed speech metrics for AMD and turn-taking.
    5. Ring buffer for smoother hysteresis and better noise rejection.
    """
    
    def __init__(self, sample_rate: int = 8000, aggressiveness: int = 1):
        """
        aggressiveness: 0 (Least aggressive) to 3 (Most aggressive). 
        Mode 1 is best for AI speech recognition - Mode 3 clips too much speech,
        cutting off faint consonants like 'H', 'S', 'F' that AI models need.
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Frame buffer for smoothing
        self.frame_duration_ms = 20  # WebRTC VAD supports 10, 20, or 30ms
        self.frame_size = int(sample_rate * (self.frame_duration_ms / 1000.0) * 2)  # bytes (16-bit)
        self.buffer = b""
        
        # Hysteresis state
        self.triggered = False
        self.speech_frames = 0
        self.silence_frames = 0
        
        # ENHANCED: Tuning parameters (professional cold calling optimized)
        self.min_speech_frames = 3   # 60ms of speech to start sending (prevents clicks)
        self.min_silence_frames = 25  # 500ms of silence to stop sending (prevents cutting off mid-sentence pauses)
        
        # ENHANCED: Ring buffer for smoother decisions (professional feature)
        self.ring_buffer_size = 10  # Track last 10 frames (200ms)
        self.ring_buffer = []  # Stores (is_speech, energy) tuples
        
        # ENHANCED: Energy tracking for adaptive systems
        self.last_frame_energy = 0.0
        self.speech_energy_sum = 0.0
        self.speech_energy_count = 0
        self.silence_energy_sum = 0.0
        self.silence_energy_count = 0
        self.current_speech_start = None
        self.current_speech_duration = 0.0
        
        # ENHANCED: Statistics for debugging and AMD
        self.total_speech_time = 0.0
        self.total_silence_time = 0.0
        self.speech_segments_count = 0
        self.last_speech_end_time = None
        
        logger.info(f"🎤 WebRTC VAD initialized (Rate={sample_rate}, Mode={aggressiveness}, SilenceFrames={self.min_silence_frames})")
    
    def _calculate_energy(self, frame: bytes) -> float:
        """Calculate RMS energy using C-based audioop"""
        try:
            # audioop.rms is significantly faster than struct.unpack + math
            return float(audioop.rms(frame, 2))
        except:
            return 0.0
    
    def process(self, audio_data: bytes) -> tuple:
        """
        Process audio chunk. Returns (is_speech, valid_audio_chunk).
        Accumulates bytes until a valid 20ms frame is ready.
        
        ENHANCED: Also tracks energy and provides metrics for adaptive systems.
        """
        self.buffer += audio_data
        
        # Process only full frames
        output_audio = b""
        has_speech = False
        
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            self.buffer = self.buffer[self.frame_size:]
            
            # ENHANCED: Calculate frame energy for adaptive systems
            frame_energy = self._calculate_energy(frame)
            self.last_frame_energy = frame_energy
            
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except:
                is_speech = True  # Fail open on error
            
            # ENHANCED: Update ring buffer for smoother decisions
            self.ring_buffer.append((is_speech, frame_energy))
            if len(self.ring_buffer) > self.ring_buffer_size:
                self.ring_buffer.pop(0)
            
            # ENHANCED: Track energy statistics
            if is_speech:
                self.speech_energy_sum += frame_energy
                self.speech_energy_count += 1
            else:
                self.silence_energy_sum += frame_energy
                self.silence_energy_count += 1
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                if self.speech_frames >= self.min_speech_frames:
                    if not self.triggered:
                        # Speech just started
                        self.current_speech_start = time.time()
                        self.speech_segments_count += 1
                    self.triggered = True
            else:
                self.silence_frames += 1
                self.speech_frames = 0
                if self.silence_frames >= self.min_silence_frames:
                    if self.triggered:
                        # Speech just ended
                        if self.current_speech_start:
                            speech_duration = time.time() - self.current_speech_start
                            self.total_speech_time += speech_duration
                            self.current_speech_duration = speech_duration
                            self.last_speech_end_time = time.time()
                        self.current_speech_start = None
                    self.triggered = False
            
            # Logic: If triggered, we pass the audio.
            if self.triggered:
                output_audio += frame
                has_speech = True
                
        return self.triggered, output_audio
    
    def process_with_metrics(self, audio_data: bytes) -> dict:
        """
        Enhanced process that returns detailed metrics for AMD and turn-taking.
        
        Returns:
            dict with keys: is_speech, audio, energy, speech_duration, 
                           avg_speech_energy, avg_silence_energy
        """
        is_speech, audio = self.process(audio_data)
        
        return {
            'is_speech': is_speech,
            'audio': audio,
            'energy': self.last_frame_energy,
            'speech_duration': self.current_speech_duration if not self.triggered else 
                              (time.time() - self.current_speech_start if self.current_speech_start else 0),
            'avg_speech_energy': self.speech_energy_sum / max(1, self.speech_energy_count),
            'avg_silence_energy': self.silence_energy_sum / max(1, self.silence_energy_count),
            'total_speech_time': self.total_speech_time,
            'speech_segments': self.speech_segments_count,
            'time_since_speech': time.time() - self.last_speech_end_time if self.last_speech_end_time else 0
        }
    
    def get_ring_buffer_speech_ratio(self) -> float:
        """Get ratio of speech frames in ring buffer (for smoothed decisions)"""
        if not self.ring_buffer:
            return 0.0
        speech_count = sum(1 for is_speech, _ in self.ring_buffer if is_speech)
        return speech_count / len(self.ring_buffer)
    
    def is_speech(self, audio_data: bytes) -> bool:
        """
        Legacy API compatibility: Simple speech detection without buffering.
        Returns True if speech is detected in this audio chunk.
        """
        try:
            # For compatibility with old code that just needs a boolean check
            if len(audio_data) < self.frame_size:
                return False
            
            # Check only the first complete frame
            frame = audio_data[:self.frame_size]
            return self.vad.is_speech(frame, self.sample_rate)
        except:
            return False
    
    def get_energy(self) -> float:
        """Get last frame's energy level"""
        return self.last_frame_energy
    
    def reset_statistics(self):
        """Reset statistics for new call"""
        self.total_speech_time = 0.0
        self.total_silence_time = 0.0
        self.speech_segments_count = 0
        self.speech_energy_sum = 0.0
        self.speech_energy_count = 0
        self.silence_energy_sum = 0.0
        self.silence_energy_count = 0
        self.ring_buffer.clear()
        self.last_speech_end_time = None
        self.current_speech_start = None


class GoodbyeDetector:
    """Detects goodbye phrases from either side and triggers graceful hangup"""
    
    POS = re.compile(r"""
        \b(
            # English
            good\s?bye|
            bye-?bye|
            thanks[, ]*bye|
            talk to you later|
            have a (nice|good) (day|one)|
            bye|

            # Spanish
            adiós|
            hasta luego|

            # French
            au revoir|
            à bientôt|

            # German
            auf Wiedersehen|
            tschüss|

            # Italian
            arrivederci|
            ciao|

            # Portuguese
            adeus|
            até logo|

            # Dutch
            tot ziens|

            # Russian
            до свидания|

            # Hindi
            alvida|

            # Japanese
            さようなら|

            # Mandarin
            再见|

            # Bulgarian
            довиждане|
            чао|
            приятен ден|
            лека нощ|
            дочуване|

            # Polish
            do widzenia|

            # Turkish
            hoşçakal|

            # Arabic
            مع السلامة|

            # Korean
            안녕히 가세요
        )\b
    """, re.I | re.VERBOSE)
    NEG = re.compile(r"(before we say goodbye|don'?t say goodbye|if you say goodbye)", re.I)

    def __init__(self, grace_ms: int = 1200):
        self.user_said = False
        self.agent_said = False
        self._timer = None
        self._grace_ms = grace_ms
        self._lock = threading.Lock()

    def feed(self, text: str, who: str, on_trigger, on_cancel):
        t = (text or "").strip()
        if not t: return
        
        # Debug logging
        logger.debug(f"🔍 Goodbye detector: checking '{t}' from {who}")
        
        if self.POS.search(t) and not self.NEG.search(t):
            logger.info(f"👋 Goodbye detected in text from {who}: '{t}'")
            with self._lock:
                if who == "user":
                    self.user_said = True
                    logger.info(f"👋 User said goodbye - arming hangup timer ({self._grace_ms}ms grace period)")
                    self._arm(on_trigger)   # grace window
                elif who == "agent":
                    # Agent goodbye detected but NOT triggering hangup - only user can end the call
                    self.agent_said = True
                    logger.info("👋 Agent said goodbye (logged only, not triggering hangup)")
        else:
            # Only cancel pending hangup if USER says something new (not agent continuing to speak)
            if self._timer and who == "user":
                logger.debug(f"🔄 Canceling goodbye timer due to new user speech: '{t}'")
                with self._lock:
                    self._disarm(on_cancel)
            elif self._timer and who == "agent":
                logger.debug(f"🔇 Agent still speaking ('{t[:50]}...'), keeping goodbye timer armed")

    def _arm(self, on_trigger, immediate=False):
        self._disarm(None)
        delay = 0.0 if immediate else self._grace_ms / 1000.0
        self._timer = threading.Timer(delay, on_trigger)
        self._timer.daemon = True
        self._timer.start()

    def _disarm(self, on_cancel):
        if self._timer:
            self._timer.cancel()
            self._timer = None
            if on_cancel: on_cancel()


class AudioPreprocessor:
    """
    Ultra-Low Latency Audio Preprocessor using pure audioop (C-based).
    Replaces heavy DSP (numpy/scipy) with simple byte manipulation.
    """
    
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        # Simple Noise Gate Threshold (RMS energy)
        # Adjust this if it cuts off quiet voices. 
        # 300-500 is usually good for 16-bit PCM.
        self.noise_threshold = 300
    
    
    
    
    
    
    def process_audio(self, audio_data: bytes) -> bytes:
        """
        Pass-through audio with simple noise gating.
        No FFT, no filtering, just raw speed.
        """
        if not audio_data:
            return b""

        try:
            # 1. Calculate Energy (RMS) using C-optimized audioop
            # width=2 means 16-bit audio
            rms = audioop.rms(audio_data, 2)
            
            # 2. Simple Noise Gate
            # If audio is too quiet, return silence to prevent static hiss 
            # from triggering the model or VAD unnecessarily.
            if rms < self.noise_threshold:
                # Return digital silence of the same length
                return b'\x00' * len(audio_data)

            # 3. Optional: Simple Limiter (Scaling)
            # If audio is clipping (too loud), scale it down slightly.
            # max_val = audioop.max(audio_data, 2)
            # if max_val > 32000:
            #     return audioop.mul(audio_data, 2, 0.9)

            return audio_data

        except Exception as e:
            logger.error(f"Error in audioop preprocessing: {e}")
            return audio_data
    
    def reset_noise_floor(self):
        pass # No state to reset in this simple version

# Import our existing voice agent components
from main import SessionLogger, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, FORMAT, CHANNELS
from google import genai
from google.genai import types

# Import CRM API
from crm_api import crm_router
from crm_auth import auth_router, get_current_user, check_subscription
from crm_user_management import user_router
from crm_superadmin import superadmin_router
from crm_billing import billing_router
from crm_database import init_database, get_session, Lead, CallSession, CallStatus, User, UserRole

# Import Gemini greeting generator (the only supported method)
try:
    from greeting_generator_gemini import generate_greeting_for_lead
    GREETING_GENERATOR_AVAILABLE = True
    GREETING_GENERATOR_METHOD = "gemini-live"
    logger.info("✅ Gemini greeting generator loaded (Gemini Live API)")
    logger.info("🎤 Using Gemini voices (Puck, Charon, Kore, Fenrir, Aoede) - same as voice calls")
except ImportError as e:
    GREETING_GENERATOR_AVAILABLE = False
    GREETING_GENERATOR_METHOD = None
    logger.warning("⚠️ Gemini greeting generator not available")
    logger.warning(f"💡 Error: {e}")
    logger.warning("💡 Make sure google-genai is installed: pip install google-genai")
except Exception as e:
    GREETING_GENERATOR_AVAILABLE = False
    GREETING_GENERATOR_METHOD = None
    logger.warning(f"⚠️ Gemini greeting generator failed to load: {e}")

# Load configuration
def load_config():
    with open('asterisk_config.json', 'r') as f:
        return json.load(f)

config = load_config()

# Forward declaration - will be initialized after app creation
sip_handler = None

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    logger.info("CRM database initialized")
    
    sip_handler.start_sip_listener()
    logger.info("Windows VoIP Voice Agent started")
    logger.info(f"Phone number: {config['phone_number']}")
    logger.info(f"Local IP: {config['local_ip']}")
    logger.info(f"Gate VoIP: {config['host']}")
    
    yield
    
    # Shutdown
    sip_handler.stop()

# Pydantic models for request bodies
class GenerateSummaryRequest(BaseModel):
    language: str = "English"

class SendSMSRequest(BaseModel):
    phone_number: str
    message: str
    gate_slot: int = 9  # Default gate slot, can be overridden

app = FastAPI(title="Windows VoIP Voice Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include CRM API routes
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(superadmin_router)
app.include_router(billing_router)
app.include_router(crm_router)

# Mount static files for serving audio recordings
from starlette.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="."), name="static")

# Global state
active_sessions: Dict[str, Dict] = {}
voice_client = genai.Client(http_options={"api_version": "v1beta"})

# Country code to language mapping
COUNTRY_LANGUAGE_MAP = {
    # European countries - Western Europe
    'BG': {'lang': 'Bulgarian', 'code': 'bg', 'formal_address': 'Вие'},
    'RO': {'lang': 'Romanian', 'code': 'ro', 'formal_address': 'Dumneavoastră'},
    'GR': {'lang': 'Greek', 'code': 'el', 'formal_address': 'Εσείς'},
    'RS': {'lang': 'Serbian', 'code': 'sr', 'formal_address': 'Ви'},
    'MK': {'lang': 'Macedonian', 'code': 'mk', 'formal_address': 'Вие'},
    'AL': {'lang': 'Albanian', 'code': 'sq', 'formal_address': 'Ju'},
    'TR': {'lang': 'Turkish', 'code': 'tr', 'formal_address': 'Siz'},
    'DE': {'lang': 'German', 'code': 'de', 'formal_address': 'Sie'},
    'IT': {'lang': 'Italian', 'code': 'it', 'formal_address': 'Lei'},
    'ES': {'lang': 'Spanish', 'code': 'es', 'formal_address': 'Usted'},
    'FR': {'lang': 'French', 'code': 'fr', 'formal_address': 'Vous'},
    'GB': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'IE': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'NL': {'lang': 'Dutch', 'code': 'nl', 'formal_address': 'U'},
    'BE': {'lang': 'Dutch', 'code': 'nl', 'formal_address': 'U'},
    'AT': {'lang': 'German', 'code': 'de', 'formal_address': 'Sie'},
    'CH': {'lang': 'German', 'code': 'de', 'formal_address': 'Sie'},
    'PL': {'lang': 'Polish', 'code': 'pl', 'formal_address': 'Pan/Pani'},
    'CZ': {'lang': 'Czech', 'code': 'cs', 'formal_address': 'Vy'},
    'SK': {'lang': 'Slovak', 'code': 'sk', 'formal_address': 'Vy'},
    'HU': {'lang': 'Hungarian', 'code': 'hu', 'formal_address': 'Ön'},
    'HR': {'lang': 'Croatian', 'code': 'hr', 'formal_address': 'Vi'},
    'SI': {'lang': 'Slovenian', 'code': 'sl', 'formal_address': 'Vi'},
    'PT': {'lang': 'Portuguese', 'code': 'pt', 'formal_address': 'Você'},
    'LU': {'lang': 'French', 'code': 'fr', 'formal_address': 'Vous'},
    'MC': {'lang': 'French', 'code': 'fr', 'formal_address': 'Vous'},
    
    # Nordic countries
    'SE': {'lang': 'Swedish', 'code': 'sv', 'formal_address': 'Ni'},
    'NO': {'lang': 'Norwegian', 'code': 'no', 'formal_address': 'Dere'},
    'DK': {'lang': 'Danish', 'code': 'da', 'formal_address': 'De'},
    'FI': {'lang': 'Finnish', 'code': 'fi', 'formal_address': 'Te'},
    'IS': {'lang': 'Icelandic', 'code': 'is', 'formal_address': 'Þið'},
    
    # Baltic countries
    'EE': {'lang': 'Estonian', 'code': 'et', 'formal_address': 'Teie'},
    'LV': {'lang': 'Latvian', 'code': 'lv', 'formal_address': 'Jūs'},
    'LT': {'lang': 'Lithuanian', 'code': 'lt', 'formal_address': 'Jūs'},
    
    # Eastern Europe
    'RU': {'lang': 'Russian', 'code': 'ru', 'formal_address': 'Вы'},
    'UA': {'lang': 'Ukrainian', 'code': 'uk', 'formal_address': 'Ви'},
    'BY': {'lang': 'Russian', 'code': 'ru', 'formal_address': 'Вы'},
    'MD': {'lang': 'Romanian', 'code': 'ro', 'formal_address': 'Dumneavoastră'},
    'MT': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'CY': {'lang': 'Greek', 'code': 'el', 'formal_address': 'Εσείς'},
    'BA': {'lang': 'Serbian', 'code': 'sr', 'formal_address': 'Vi'},
    'ME': {'lang': 'Serbian', 'code': 'sr', 'formal_address': 'Vi'},
    'XK': {'lang': 'Albanian', 'code': 'sq', 'formal_address': 'Ju'},
    
    # North America
    'US': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'CA': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'MX': {'lang': 'Spanish', 'code': 'es', 'formal_address': 'Usted'},
    
    # Oceania
    'AU': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'NZ': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    
    # Asia - East Asia
    'CN': {'lang': 'Chinese', 'code': 'zh', 'formal_address': '您'},
    'JP': {'lang': 'Japanese', 'code': 'ja', 'formal_address': 'あなた'},
    'KR': {'lang': 'Korean', 'code': 'ko', 'formal_address': '당신'},
    'TW': {'lang': 'Chinese', 'code': 'zh-TW', 'formal_address': '您'},
    'HK': {'lang': 'Chinese', 'code': 'zh-HK', 'formal_address': '您'},
    'SG': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'MY': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    
    # Asia - South/Southeast Asia
    'IN': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'PK': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'BD': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'TH': {'lang': 'Thai', 'code': 'th', 'formal_address': 'คุณ'},
    'VN': {'lang': 'Vietnamese', 'code': 'vi', 'formal_address': 'Anh/Chị'},
    'PH': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'ID': {'lang': 'Indonesian', 'code': 'id', 'formal_address': 'Anda'},
    
    # Latin America - South America
    'BR': {'lang': 'Portuguese', 'code': 'pt-BR', 'formal_address': 'Você'},
    'AR': {'lang': 'Spanish', 'code': 'es-AR', 'formal_address': 'Usted'},
    'CL': {'lang': 'Spanish', 'code': 'es-CL', 'formal_address': 'Usted'},
    'CO': {'lang': 'Spanish', 'code': 'es-CO', 'formal_address': 'Usted'},
    'PE': {'lang': 'Spanish', 'code': 'es-PE', 'formal_address': 'Usted'},
    'VE': {'lang': 'Spanish', 'code': 'es-VE', 'formal_address': 'Usted'},
    'UY': {'lang': 'Spanish', 'code': 'es-UY', 'formal_address': 'Usted'},
    'PY': {'lang': 'Spanish', 'code': 'es-PY', 'formal_address': 'Usted'},
    'EC': {'lang': 'Spanish', 'code': 'es-EC', 'formal_address': 'Usted'},
    'BO': {'lang': 'Spanish', 'code': 'es-BO', 'formal_address': 'Usted'},
    
    # Latin America - Central America & Caribbean
    'GT': {'lang': 'Spanish', 'code': 'es-GT', 'formal_address': 'Usted'},
    'CR': {'lang': 'Spanish', 'code': 'es-CR', 'formal_address': 'Usted'},
    'PA': {'lang': 'Spanish', 'code': 'es-PA', 'formal_address': 'Usted'},
    'NI': {'lang': 'Spanish', 'code': 'es-NI', 'formal_address': 'Usted'},
    'HN': {'lang': 'Spanish', 'code': 'es-HN', 'formal_address': 'Usted'},
    'SV': {'lang': 'Spanish', 'code': 'es-SV', 'formal_address': 'Usted'},
    'BZ': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'CU': {'lang': 'Spanish', 'code': 'es-CU', 'formal_address': 'Usted'},
    'DO': {'lang': 'Spanish', 'code': 'es-DO', 'formal_address': 'Usted'},
    'PR': {'lang': 'Spanish', 'code': 'es-PR', 'formal_address': 'Usted'},
    'JM': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    
    # Middle East
    'AE': {'lang': 'Arabic', 'code': 'ar', 'formal_address': 'أنت'},
    'SA': {'lang': 'Arabic', 'code': 'ar-SA', 'formal_address': 'أنت'},
    'QA': {'lang': 'Arabic', 'code': 'ar-QA', 'formal_address': 'أنت'},
    'KW': {'lang': 'Arabic', 'code': 'ar-KW', 'formal_address': 'أنت'},
    'BH': {'lang': 'Arabic', 'code': 'ar-BH', 'formal_address': 'أنت'},
    'OM': {'lang': 'Arabic', 'code': 'ar-OM', 'formal_address': 'أنت'},
    'JO': {'lang': 'Arabic', 'code': 'ar-JO', 'formal_address': 'أنت'},
    'LB': {'lang': 'Arabic', 'code': 'ar-LB', 'formal_address': 'أنت'},
    'SY': {'lang': 'Arabic', 'code': 'ar-SY', 'formal_address': 'أنت'},
    'IQ': {'lang': 'Arabic', 'code': 'ar-IQ', 'formal_address': 'أنت'},
    'IL': {'lang': 'Hebrew', 'code': 'he', 'formal_address': 'אתה/את'},
    'IR': {'lang': 'Persian', 'code': 'fa', 'formal_address': 'شما'},
    'AF': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    
    # Africa
    'ZA': {'lang': 'English', 'code': 'en-ZA', 'formal_address': 'You'},
    'NG': {'lang': 'English', 'code': 'en-NG', 'formal_address': 'You'},
    'EG': {'lang': 'Arabic', 'code': 'ar-EG', 'formal_address': 'أنت'},
    'KE': {'lang': 'English', 'code': 'en-KE', 'formal_address': 'You'},
    'GH': {'lang': 'English', 'code': 'en-GH', 'formal_address': 'You'},
    'TN': {'lang': 'Arabic', 'code': 'ar-TN', 'formal_address': 'أنت'},
    'MA': {'lang': 'Arabic', 'code': 'ar-MA', 'formal_address': 'أنت'},
    'DZ': {'lang': 'Arabic', 'code': 'ar-DZ', 'formal_address': 'أنت'},
    'ET': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'UG': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
    'TZ': {'lang': 'English', 'code': 'en', 'formal_address': 'You'},
}

# Phone number prefixes to country mapping (simplified)
PHONE_COUNTRY_MAP = {
    # Europe - Western Europe
    '+359': 'BG', '359': 'BG',  # Bulgaria
    '+40': 'RO', '40': 'RO',    # Romania  
    '+30': 'GR', '30': 'GR',    # Greece
    '+381': 'RS', '381': 'RS',  # Serbia
    '+389': 'MK', '389': 'MK',  # North Macedonia
    '+355': 'AL', '355': 'AL',  # Albania
    '+90': 'TR', '90': 'TR',    # Turkey
    '+49': 'DE', '49': 'DE',    # Germany
    '+39': 'IT', '39': 'IT',    # Italy
    '+34': 'ES', '34': 'ES',    # Spain
    '+33': 'FR', '33': 'FR',    # France
    '+44': 'GB', '44': 'GB',    # UK
    '+353': 'IE', '353': 'IE',  # Ireland
    '+31': 'NL', '31': 'NL',    # Netherlands
    '+32': 'BE', '32': 'BE',    # Belgium
    '+43': 'AT', '43': 'AT',    # Austria
    '+41': 'CH', '41': 'CH',    # Switzerland
    '+48': 'PL', '48': 'PL',    # Poland
    '+420': 'CZ', '420': 'CZ',  # Czech Republic
    '+421': 'SK', '421': 'SK',  # Slovakia
    '+36': 'HU', '36': 'HU',    # Hungary
    '+385': 'HR', '385': 'HR',  # Croatia
    '+386': 'SI', '386': 'SI',  # Slovenia
    '+351': 'PT', '351': 'PT',  # Portugal
    '+352': 'LU', '352': 'LU',  # Luxembourg
    '+377': 'MC', '377': 'MC',  # Monaco
    
    # Nordic countries
    '+46': 'SE', '46': 'SE',    # Sweden
    '+47': 'NO', '47': 'NO',    # Norway
    '+45': 'DK', '45': 'DK',    # Denmark
    '+358': 'FI', '358': 'FI',  # Finland
    '+354': 'IS', '354': 'IS',  # Iceland
    
    # Baltic countries
    '+372': 'EE', '372': 'EE',  # Estonia
    '+371': 'LV', '371': 'LV',  # Latvia
    '+370': 'LT', '370': 'LT',  # Lithuania
    
    # Eastern Europe
    '+7': 'RU', '7': 'RU',      # Russia (also Kazakhstan)
    '+380': 'UA', '380': 'UA',  # Ukraine
    '+375': 'BY', '375': 'BY',  # Belarus
    '+373': 'MD', '373': 'MD',  # Moldova
    '+356': 'MT', '356': 'MT',  # Malta
    '+357': 'CY', '357': 'CY',  # Cyprus
    '+387': 'BA', '387': 'BA',  # Bosnia and Herzegovina
    '+382': 'ME', '382': 'ME',  # Montenegro
    '+383': 'XK', '383': 'XK',  # Kosovo
    
    # North America
    '+1': 'US', '1': 'US',      # USA/Canada (will differentiate by area code later)
    '+52': 'MX', '52': 'MX',    # Mexico
    
    # Oceania
    '+61': 'AU', '61': 'AU',    # Australia
    '+64': 'NZ', '64': 'NZ',    # New Zealand
    
    # Asia - East Asia
    '+86': 'CN', '86': 'CN',    # China
    '+81': 'JP', '81': 'JP',    # Japan
    '+82': 'KR', '82': 'KR',    # South Korea
    '+886': 'TW', '886': 'TW',  # Taiwan
    '+852': 'HK', '852': 'HK',  # Hong Kong
    '+65': 'SG', '65': 'SG',    # Singapore
    '+60': 'MY', '60': 'MY',    # Malaysia
    
    # Asia - South/Southeast Asia
    '+91': 'IN', '91': 'IN',    # India
    '+92': 'PK', '92': 'PK',    # Pakistan
    '+880': 'BD', '880': 'BD',  # Bangladesh
    '+66': 'TH', '66': 'TH',    # Thailand
    '+84': 'VN', '84': 'VN',    # Vietnam
    '+63': 'PH', '63': 'PH',    # Philippines
    '+62': 'ID', '62': 'ID',    # Indonesia
    
    # Latin America - South America
    '+55': 'BR', '55': 'BR',    # Brazil
    '+54': 'AR', '54': 'AR',    # Argentina
    '+56': 'CL', '56': 'CL',    # Chile
    '+57': 'CO', '57': 'CO',    # Colombia
    '+51': 'PE', '51': 'PE',    # Peru
    '+58': 'VE', '58': 'VE',    # Venezuela
    '+598': 'UY', '598': 'UY',  # Uruguay
    '+595': 'PY', '595': 'PY',  # Paraguay
    '+593': 'EC', '593': 'EC',  # Ecuador
    '+591': 'BO', '591': 'BO',  # Bolivia
    
    # Latin America - Central America & Caribbean
    '+502': 'GT', '502': 'GT',  # Guatemala
    '+506': 'CR', '506': 'CR',  # Costa Rica
    '+507': 'PA', '507': 'PA',  # Panama
    '+505': 'NI', '505': 'NI',  # Nicaragua
    '+504': 'HN', '504': 'HN',  # Honduras
    '+503': 'SV', '503': 'SV',  # El Salvador
    '+501': 'BZ', '501': 'BZ',  # Belize
    '+53': 'CU', '53': 'CU',    # Cuba
    '+1809': 'DO', '1809': 'DO', # Dominican Republic
    '+1829': 'DO', '1829': 'DO', # Dominican Republic (alt)
    '+1849': 'DO', '1849': 'DO', # Dominican Republic (alt)
    '+1787': 'PR', '1787': 'PR', # Puerto Rico
    '+1939': 'PR', '1939': 'PR', # Puerto Rico (alt)
    '+1876': 'JM', '1876': 'JM', # Jamaica
    
    # Middle East
    '+971': 'AE', '971': 'AE',  # UAE
    '+966': 'SA', '966': 'SA',  # Saudi Arabia
    '+974': 'QA', '974': 'QA',  # Qatar
    '+965': 'KW', '965': 'KW',  # Kuwait
    '+973': 'BH', '973': 'BH',  # Bahrain
    '+968': 'OM', '968': 'OM',  # Oman
    '+962': 'JO', '962': 'JO',  # Jordan
    '+961': 'LB', '961': 'LB',  # Lebanon
    '+963': 'SY', '963': 'SY',  # Syria
    '+964': 'IQ', '964': 'IQ',  # Iraq
    '+972': 'IL', '972': 'IL',  # Israel
    '+98': 'IR', '98': 'IR',    # Iran
    '+93': 'AF', '93': 'AF',    # Afghanistan
    
    # Africa
    '+27': 'ZA', '27': 'ZA',    # South Africa
    '+234': 'NG', '234': 'NG',  # Nigeria
    '+20': 'EG', '20': 'EG',    # Egypt
    '+254': 'KE', '254': 'KE',  # Kenya
    '+233': 'GH', '233': 'GH',  # Ghana
    '+216': 'TN', '216': 'TN',  # Tunisia
    '+212': 'MA', '212': 'MA',  # Morocco
    '+213': 'DZ', '213': 'DZ',  # Algeria
    '+251': 'ET', '251': 'ET',  # Ethiopia
    '+256': 'UG', '256': 'UG',  # Uganda
    '+255': 'TZ', '255': 'TZ',  # Tanzania
}

def detect_caller_country(phone_number: str) -> str:
    """
    Detect caller's country from phone number.
    Returns country code (e.g., 'BG', 'RO') or 'BG' as default.
    """
    if not phone_number or phone_number == "Unknown":
        logger.info(f"📍 No phone number provided, defaulting to Bulgaria")
        return 'BG'  # Default to Bulgaria
    
    logger.debug(f"📍 Analyzing phone number: {phone_number}")
    
    # Clean the phone number - keep + and digits only
    clean_number = re.sub(r'[^0-9+]', '', phone_number)
    
    # Remove SIM gate prefix (single digit 9 at the start when followed by country code)
    # This handles cases like "9359..." where 9 is the gate port, not part of the actual number
    # Note: We need to be careful not to strip legitimate country codes like 91 (India)
    if clean_number.startswith('9') and len(clean_number) > 10:
        # Check if removing the 9 would give us a recognizable country code
        without_prefix = clean_number[1:]
        # Check if the number without '9' matches any known country prefix
        has_valid_prefix = False
        for country_prefix in PHONE_COUNTRY_MAP.keys():
            # Remove the '+' from the prefix for comparison
            numeric_prefix = country_prefix.replace('+', '')
            if without_prefix.startswith(numeric_prefix):
                has_valid_prefix = True
                break
        
        if has_valid_prefix:
            logger.debug(f"📍 Detected SIM gate prefix '9' - stripping it: {clean_number} → {without_prefix}")
            clean_number = without_prefix
    
    # Handle different number formats
    # If it starts with 00, replace with +
    if clean_number.startswith('00'):
        clean_number = '+' + clean_number[2:]
    
    # If it doesn't start with + and is long enough, try adding +
    elif not clean_number.startswith('+') and len(clean_number) > 7:
        clean_number = '+' + clean_number
    
    logger.debug(f"📍 Cleaned phone number: {clean_number}")
    
    # Check for exact matches first (longer prefixes first)
    sorted_prefixes = sorted(PHONE_COUNTRY_MAP.keys(), key=len, reverse=True)
    
    for prefix in sorted_prefixes:
        if clean_number.startswith(prefix):
            country = PHONE_COUNTRY_MAP[prefix]
            logger.info(f"📍 ✅ Detected country {country} from phone number {phone_number} → {clean_number} (matched prefix: {prefix})")
            return country
    
    # If no match found, default to Bulgaria
    logger.warning(f"📍 ⚠️ Could not detect country from phone number {phone_number} → {clean_number}, defaulting to Bulgaria")
    return 'BG'

def get_language_config(country_code: str) -> Dict[str, Any]:
    """
    Get language configuration for a country.
    Returns language info or Bulgarian as default.
    """
    return COUNTRY_LANGUAGE_MAP.get(country_code, COUNTRY_LANGUAGE_MAP['BG'])

def create_voice_config(language_info: Dict[str, Any], custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create dynamic voice configuration based on detected language and custom parameters.
    All instructions now come from the CRM configuration - no hardcoded content.
    """
    lang_name = language_info['lang']
    formal_address = language_info['formal_address']
    
    # Use custom config if provided, otherwise use defaults
    if custom_config:
        company_name = custom_config.get('company_name', 'PropTechAI')
        caller_name = custom_config.get('caller_name', 'Assistant')
        product_name = custom_config.get('product_name', 'our product')
        additional_prompt = custom_config.get('additional_prompt', '')
        call_urgency = custom_config.get('call_urgency', 'medium')
        call_objective = custom_config.get('call_objective', 'sales')
        main_benefits = custom_config.get('main_benefits', '')
        special_offer = custom_config.get('special_offer', '')
        objection_strategy = custom_config.get('objection_strategy', 'understanding')
        greeting_transcript = custom_config.get('greeting_transcript', '')  # ✅ Get greeting text
        voice_name = custom_config.get('voice_name', 'Puck')  # ✅ Get voice name from CRM
        gemini_greeting = custom_config.get('gemini_greeting', False)  # ✅ Let Gemini speak the greeting
        greeting_instruction = custom_config.get('greeting_instruction', '')  # ✅ Custom greeting text
    else:
        # Minimal defaults when no custom config is provided
        company_name = 'PropTechAI'
        caller_name = 'Assistant'
        product_name = 'our product'
        additional_prompt = ''
        call_urgency = 'medium'
        call_objective = 'sales'
        main_benefits = ''
        special_offer = ''
        objection_strategy = 'understanding'
        greeting_transcript = ''  # ✅ No greeting by default
        voice_name = 'Puck'  # ✅ Default voice
        gemini_greeting = False  # ✅ Default: use file-based greeting
        greeting_instruction = ''  # ✅ No custom greeting
    
    # Create system instruction in the detected language
    if lang_name == 'English':
        system_text = f"You are {caller_name} from {company_name}, a professional sales representative for {product_name}. "
        
        # ✅ Handle greeting - either Gemini-initiated or pre-recorded
        if gemini_greeting:
            # Gemini starts the conversation with a greeting
            if greeting_instruction:
                system_text += f"CRITICAL: You MUST start the conversation IMMEDIATELY by speaking first. Say this greeting: \"{greeting_instruction}\". Do NOT wait for the caller to speak first. After you greet them, pause briefly and wait for their response. "
            else:
                system_text += f"CRITICAL: You MUST start the conversation IMMEDIATELY by speaking first. Introduce yourself briefly: say hello, your name ({caller_name}), and that you're calling from {company_name}. Do NOT wait for the caller to speak first. After your brief introduction, pause and wait for their response. "
        elif greeting_transcript:
            # Pre-recorded greeting was already played
            system_text += f"IMPORTANT: You have ALREADY played this greeting to the caller: \"{greeting_transcript}\". DO NOT repeat this greeting. DO NOT introduce yourself again. The caller has already heard your introduction. Wait for the caller to speak first, then respond naturally to what they say. "
        
        # Add objective-specific instructions
        if call_objective == "sales":
            system_text += "You are making sales calls to sell this product. Focus on converting prospects into customers by highlighting product benefits and closing the sale. "
        elif call_objective == "followup":
            system_text += "You are following up on a previous interaction. Be friendly and check on their interest while guiding toward a purchase decision. "
        elif call_objective == "survey":
            system_text += "You are conducting a survey but also identifying sales opportunities. Ask relevant questions while presenting the product benefits. "
        elif call_objective == "appointment":
            system_text += "You are cold calling to set appointments or qualify leads. Focus on building rapport, understanding their needs, and scheduling a follow-up meeting or call. "
        elif call_objective == "ai_sales_services":
            system_text += """You are making a professional cold call to a real estate agent to sell AI automation solutions. Follow this structure:

1. INTRODUCTION & PATTERN INTERRUPT (first 15 seconds):
   - Introduce yourself and the company
   - Mention the goal - real estate automation with AI
   - Ask: "Did I catch you at a bad time?"

2. PRESENT THE PROBLEM (if they say "I have a minute"):
   - Explain the problem: many agents lose potential clients because they can't respond to everyone immediately
   - Missed calls after hours and on weekends
   - Time spent writing property descriptions
   - Unqualified buyers taking up time
   - Manual scheduling of viewings

3. PRESENT THE SOLUTION:
   - AI assistant that answers inquiries 24/7
   - Automatically generates professional property descriptions
   - Intelligent buyer qualification (budget, requirements, seriousness)
   - Automated viewing scheduling
   - Frees up time for actual sales
   - Can handle 10x more inquiries without hiring staff

4. QUALIFYING QUESTIONS:
   - "How many property inquiries do you get weekly?"
   - "Are you able to respond to all of them or do you lose potential clients?"
   - "Do you have an assistant or work alone?"
   - "How much time do you spend writing descriptions and coordinating viewings?"

5. SCHEDULE A MEETING:
   - Mention social proof: "We work with leading agents in Sofia who've already increased their inquiries by 300%"
   - Suggest a 15-minute demo
   - SPECIAL OFFER: "The first 30 days are completely free for early clients"
   - Give specific options: "Tomorrow afternoon or Friday morning - which works better?"
   - IF THE CLIENT SHOWS INTEREST AND YOU AGREE ON A MEETING: Remember the day and time and mention them at the end for confirmation

6. HANDLING OBJECTIONS:
   - "Too expensive" → "I understand the concern. Actually, the first 30 days are free. After that, it's less than one month's assistant salary, and it works 24/7. How much is one property sale worth?"
   - "Already have an assistant" → "Great! Our AI actually helps your assistant be more effective. It handles routine inquiries while your assistant focuses on real clients. Can we show you how in 15 minutes?"
   - "No time now" → "That's exactly why you need AI - to free up time! The demo is only 15 minutes and will show you how to save hours every week"
   - "Need to think" → "Of course. Just keep in mind that the special offer for the first 30 days free is only for a limited number of agencies. Could we reserve a spot for you with a quick demo?"

IMPORTANT: 
- Be professional, direct, and focused on value
- Speak the agents' language - time, money, sales
- Emphasize that this is not an expense but an investment that pays back many times over
- If the client agrees to a meeting, ask for a convenient day and time and remember them
- Focus on specific real estate pain points """
        elif call_objective == "companions_services":
            system_text += """You are calling from a premium escort agency. Be professional, discreet, and polite. Follow this structure:

1. DISCREET INTRODUCTION:
   - Introduce yourself professionally with name and agency
   - Be polite and respect their personal space
   - Ask if it's a convenient time to talk

2. SERVICE PRESENTATION:
   - Mention that you offer premium escort services
   - Emphasize professionalism, discretion, and elegance
   - Explain that you have educated and attractive ladies
   - Services include event companionship, meetings, VIP arrangements

3. KEY BENEFITS:
   - Full discretion and confidentiality guaranteed
   - Flexible scheduling (24/7 availability)
   - Safety and professional behavior
   - High level of service

4. APPOINTMENT SCHEDULING:
   - Ask if they are interested
   - Suggest a meeting tonight or tomorrow
   - Mention the 20% discount for first meeting
   - Offer to send portfolio discreetly (photos of ladies)
   - IF YOU AGREE ON A MEETING: Remember the time and day

5. HANDLING OBJECTIONS:
   - "Not sure" → Emphasize discretion and professionalism. Offer to send information without obligation
   - "Too expensive" → Explain the quality of service and mention the first-time discount
   - "No time" → "I understand. We work 24/7, we can arrange a meeting whenever convenient for you"
   - "Married/In a relationship" → Assure them of complete discretion and confidentiality

IMPORTANT: 
- Be professional, polite, and respect personal boundaries
- NEVER be vulgar or rude
- Emphasize discretion and safety
- Respect refusal - if the client is not interested, thank them politely and end the call
- Speak naturally and confidently, without sounding like a salesperson """
        
        # Add main benefits if provided
        if main_benefits:
            system_text += f"Key benefits to emphasize: {main_benefits}. "
        
        # Add special offers if provided
        if special_offer:
            system_text += f"Current offers: {special_offer}. "
        
        # Add urgency based on setting
        if call_urgency == "high":
            system_text += "Create MAXIMUM URGENCY - emphasize time-sensitive offers and limited availability. "
        elif call_urgency == "medium":
            system_text += "Create moderate urgency with special offers and time-sensitive deals. "
        else:
            system_text += "Be persistent but not overly aggressive. Focus on building rapport and trust. "
        
        # Add objection handling strategy
        if objection_strategy == "understanding":
            system_text += "Handle objections with empathy and understanding. Listen to their concerns and address them thoughtfully. "
        elif objection_strategy == "educational":
            system_text += "Handle objections by providing educational information and facts to overcome doubts. "
        elif objection_strategy == "aggressive":
            system_text += "Handle objections persistently. Push back on concerns and maintain strong sales pressure. "
        
        system_text += f"Always maintain professionalism and use formal address ({formal_address}). Speak clearly, enthusiastically, and with confidence."
        
        # Add technical context about audio quality
        system_text += " You are speaking over a legacy telephone line. "
        system_text += "The audio quality may be low, muffled, or contain static. "
        system_text += "If you cannot clearly understand what the user said, DO NOT GUESS. "
        system_text += "Instead, politely ask them to repeat themselves or ask 'I'm sorry, the connection is bad, could you say that again?' "
        system_text += "Do not hallucinate answers based on static noise."
        
        # Add additional prompt if provided
        if additional_prompt:
            system_text += f" Additional instructions: {additional_prompt}"
            
    elif lang_name == 'Bulgarian':
        system_text = f"Вие сте {caller_name} от {company_name}, професионален търговски представител на {product_name}. "
        
        # ✅ Handle greeting - either Gemini-initiated or pre-recorded (in Bulgarian)
        if gemini_greeting:
            # Gemini starts the conversation with a greeting
            if greeting_instruction:
                system_text += f"КРИТИЧНО: ТРЯБВА да започнете разговора ВЕДНАГА, като говорите първи. Кажете това поздравление: \"{greeting_instruction}\". НЕ чакайте обаждащият се да говори пръв. След като го поздравите, направете кратка пауза и изчакайте отговора му. "
            else:
                system_text += f"КРИТИЧНО: ТРЯБВА да започнете разговора ВЕДНАГА, като говорите първи. Представете се кратко: поздравете, кажете името си ({caller_name}) и че се обаждате от {company_name}. НЕ чакайте обаждащият се да говори пръв. След краткото си представяне, направете пауза и изчакайте отговора му. "
        elif greeting_transcript:
            # Pre-recorded greeting was already played
            system_text += f"ВАЖНО: Вече сте изпратили това приветствие на обаждащия се: \"{greeting_transcript}\". НЕ повтаряйте това приветствие. НЕ се представяйте отново. Обаждащият се вече е чул вашето представяне. Изчакайте обаждащият се първо да говори, след това отговорете естествено на това, което казва. "
        
        # Add objective-specific instructions in Bulgarian
        if call_objective == "sales":
            system_text += "Вие правите търговски обаждания за продажба на този продукт. Фокусирайте се върху превръщането на потенциалните клиенти в купувачи, като подчертавате ползите от продукта и затваряте продажбата. "
        elif call_objective == "followup":
            system_text += "Вие се обаждате за проследяване на предишно взаимодействие. Бъдете приятелски настроен и проверете интереса им, като ги насочите към решение за покупка. "
        elif call_objective == "survey":
            system_text += "Вие провеждате проучване, но също така търсите възможности за продажби. Задавайте подходящи въпроси, като представяте ползите от продукта. "
        elif call_objective == "appointment":
            system_text += "Вие правите студени обаждания за назначаване на срещи или квалифициране на потенциални клиенти. Фокусирайте се върху изграждането на връзка, разбирането на техните нужди и планирането на последваща среща или обаждане. "
        elif call_objective == "ai_sales_services":
            system_text += """Вие продавате AI решения за автоматизация на брокери на недвижими имоти. Говорете професионално и директно.

СТИЛ НА ГОВОРЕНЕ:
- Давайте ИНФОРМАЦИЯ, не задавайте много въпроси
- Описвайте решението и ползите ясно
- Максимум 1 въпрос на всеки 3-4 изречения  
- Бъдете кратки (2-3 изречения на отговор)
- Нека клиентът говори, Вие слушайте и отговаряйте

КАКВО ДА КАЖЕТЕ:
- Представете се: име, компания, AI за недвижими имоти
- Опишете решението: AI асистент 24/7, генериране на описания, квалифициране на купувачи, 10x повече запитвания
- Споменете: Работим с водещи брокери в София (300% увеличение)
- Оферта: Първите 30 дни безплатни
- Предложете 15-минутна демонстрация - утре или петък
- При интерес - договорете час и запомнете

СПРАВЯНЕ С ВЪЗРАЖЕНИЯ:
- "Скъпо" → По-евтино от асистент, работи 24/7, плюс 30 дни безплатно
- "Имам асистент" → AI помага на асистента, поема рутината
- "Няма време" → Точно затова ви трябва - спестява часове. Демото 15 мин
- "Трябва да помисля" → Разбира се. Офертата е за ограничен брой агенции

ВАЖНО:
- Говорете на езика на брокерите: време, пари, продажби
- Това е инвестиция, не разход
- Фокус върху болките им - пропуснати обаждания, време за писане, неквалифицирани купувачи
- Слушайте повече, говорете по-малко """
        elif call_objective == "companions_services":
            system_text += """Вие се обаждате от премиум ескорт агенция. Бъдете професионален, дискретен и естествен в разговора.

СТИЛ НА ГОВОРЕНЕ:
- Давайте ИНФОРМАЦИЯ и предложения, не задавайте много въпроси
- Говорете с увереност и описвайте услугите ясно
- Нека клиентът задава въпросите - Вие отговаряйте
- Максимум 1 въпрос на всеки 3-4 изречения
- Бъдете кратки и директни (2-3 изречения на отговор)

КАКВО ДА КАЖЕТЕ:
- Представете се кратко: име, агенция, какво предлагате
- Опишете услугата: премиум ескорт, дискретност, елегантни дами, 24/7
- Споменете портфолио и 20% отстъпка за първа среща
- Предложете конкретно време за среща
- При интерес - договорете час и запомнете го

СПРАВЯНЕ С ВЪЗРАЖЕНИЯ:
- "Не съм сигурен" → Няма проблем. Мога да изпратя портфолио дискретно
- "Скъпо" → Качеството си има цена, плюс 20% отстъпка първи път
- "Няма време" → Работим 24/7, удобно време за Вас
- "Женен" → Пълна дискретност гарантирана

ВАЖНО:
- Бъдете учтив, професионален, никога вулгарен
- АКО клиентът не е заинтересован - благодарете и прекратете
- Говорете естествено, НЕ като робот или продавач
- Очаквайте клиентът да говори, после отговаряйте кратко """
        
        # Add main benefits if provided
        if main_benefits:
            system_text += f"Ключови ползи за подчертаване: {main_benefits}. "
        
        # Add special offers if provided
        if special_offer:
            system_text += f"Текущи оферти: {special_offer}. "
        
        # Add urgency based on setting
        if call_urgency == "high":
            system_text += "Създавайте МАКСИМАЛНА НЕОТЛОЖНОСТ - подчертавайте офертите, ограничени по време, и ограничената наличност. "
        elif call_urgency == "medium":
            system_text += "Създавайте умерена неотложност със специални оферти и сделки, ограничени по време. "
        else:
            system_text += "Бъдете настоятелен, но не прекалено агресивен. Фокусирайте се върху изграждането на връзка и доверие. "
        
        # Add objection handling strategy
        if objection_strategy == "understanding":
            system_text += "Справяйте се с възраженията с емпатия и разбиране. Слушайте загриженостите им и ги адресирайте внимателно. "
        elif objection_strategy == "educational":
            system_text += "Справяйте се с възраженията, като предоставяте образователна информация и факти за преодоляване на съмненията. "
        elif objection_strategy == "aggressive":
            system_text += "Справяйте се с възраженията настоятелно. Противопоставете се на загриженостите и поддържайте силно търговско напрежение. "
        
        system_text += f"Винаги поддържайте професионализъм и използвайте формално обръщение ({formal_address}). Говорете ясно, ентусиазирано и с увереност. "
        
        # Critical: reduce excessive questioning
        system_text += "ВАЖНО ЗА СТИЛ: Давайте повече ИНФОРМАЦИЯ и по-малко ВЪПРОСИ. Нека клиентът задава въпросите. Вие отговаряйте кратко и ясно (2-3 изречения). Максимум 1 въпрос на всеки 3-4 изречения. Говорете естествено, НЕ като робот."
        
        # Add additional prompt if provided
        if additional_prompt:
            system_text += f" Допълнителни инструкции: {additional_prompt}"
            
    else:
        # For other languages, use English template but mention the language
        system_text = f"You are {caller_name} from {company_name}, a professional sales representative for {product_name}, speaking in {lang_name}. "
        
        # ✅ Handle greeting - either Gemini-initiated or pre-recorded
        if gemini_greeting:
            # Gemini starts the conversation with a greeting
            if greeting_instruction:
                system_text += f"CRITICAL: You MUST start the conversation IMMEDIATELY by speaking first in {lang_name}. Say this greeting: \"{greeting_instruction}\". Do NOT wait for the caller to speak first. After you greet them, pause briefly and wait for their response. "
            else:
                system_text += f"CRITICAL: You MUST start the conversation IMMEDIATELY by speaking first in {lang_name}. Introduce yourself briefly: say hello, your name ({caller_name}), and that you're calling from {company_name}. Do NOT wait for the caller to speak first. After your brief introduction, pause and wait for their response. "
        elif greeting_transcript:
            # Pre-recorded greeting was already played
            system_text += f"IMPORTANT: You have ALREADY played this greeting to the caller: \"{greeting_transcript}\". DO NOT repeat this greeting. DO NOT introduce yourself again. The caller has already heard your introduction. Wait for the caller to speak first, then respond naturally to what they say. "
        
        # Add objective-specific instructions
        if call_objective == "sales":
            system_text += "You are making sales calls to sell this product. Focus on converting prospects into customers by highlighting product benefits and closing the sale. "
        elif call_objective == "followup":
            system_text += "You are following up on a previous interaction. Be friendly and check on their interest while guiding toward a purchase decision. "
        elif call_objective == "survey":
            system_text += "You are conducting a survey but also identifying sales opportunities. Ask relevant questions while presenting the product benefits. "
        elif call_objective == "appointment":
            system_text += "You are cold calling to set appointments or qualify leads. Focus on building rapport, understanding their needs, and scheduling a follow-up meeting or call. "
        elif call_objective == "ai_sales_services":
            system_text += f"""You are making a professional cold call to a real estate agent in {lang_name} to sell AI automation solutions. Follow this structure:

1. INTRODUCTION & PATTERN INTERRUPT (first 15 seconds):
   - Introduce yourself and the company
   - Mention the goal - real estate automation with AI
   - Ask: "Did I catch you at a bad time?"

2. PRESENT THE PROBLEM (if they say "I have a minute"):
   - Explain the problem: many agents lose potential clients because they can't respond to everyone immediately
   - Missed calls after hours and on weekends
   - Time spent writing property descriptions
   - Unqualified buyers taking up time
   - Manual scheduling of viewings

3. PRESENT THE SOLUTION:
   - AI assistant that answers inquiries 24/7
   - Automatically generates professional property descriptions
   - Intelligent buyer qualification (budget, requirements, seriousness)
   - Automated viewing scheduling
   - Frees up time for actual sales
   - Can handle 10x more inquiries without hiring staff

4. QUALIFYING QUESTIONS:
   - "How many property inquiries do you get weekly?"
   - "Are you able to respond to all of them or do you lose potential clients?"
   - "Do you have an assistant or work alone?"
   - "How much time do you spend writing descriptions and coordinating viewings?"

5. SCHEDULE A MEETING:
   - Mention social proof: "We work with leading agents who've already increased their inquiries by 300%"
   - Suggest a 15-minute demo
   - SPECIAL OFFER: "The first 30 days are completely free for early clients"
   - Give specific options: "Tomorrow afternoon or Friday morning - which works better?"
   - IF THE CLIENT SHOWS INTEREST AND YOU AGREE ON A MEETING: Remember the day and time and mention them at the end for confirmation

6. HANDLING OBJECTIONS:
   - "Too expensive" → "I understand the concern. Actually, the first 30 days are free. After that, it's less than one month's assistant salary, and it works 24/7. How much is one property sale worth?"
   - "Already have an assistant" → "Great! Our AI actually helps your assistant be more effective. It handles routine inquiries while your assistant focuses on real clients. Can we show you how in 15 minutes?"
   - "No time now" → "That's exactly why you need AI - to free up time! The demo is only 15 minutes and will show you how to save hours every week"
   - "Need to think" → "Of course. Just keep in mind that the special offer for the first 30 days free is only for a limited number of agencies. Could we reserve a spot for you with a quick demo?"

IMPORTANT: 
- Be professional, direct, and focused on value
- Speak the agents' language - time, money, sales
- Emphasize that this is not an expense but an investment that pays back many times over
- If the client agrees to a meeting, ask for a convenient day and time and remember them
- Focus on specific real estate pain points """
        elif call_objective == "companions_services":
            system_text += f"""You are calling from a premium escort agency, speaking in {lang_name}. Be professional, discreet, and polite. Follow this structure:

1. DISCREET INTRODUCTION:
   - Introduce yourself professionally with name and agency
   - Be polite and respect their personal space
   - Ask if it's a convenient time to talk

2. SERVICE PRESENTATION:
   - Mention that you offer premium escort services
   - Emphasize professionalism, discretion, and elegance
   - Explain that you have educated and attractive ladies
   - Services include event companionship, meetings, VIP arrangements

3. KEY BENEFITS:
   - Full discretion and confidentiality guaranteed
   - Flexible scheduling (24/7 availability)
   - Safety and professional behavior
   - High level of service

4. APPOINTMENT SCHEDULING:
   - Ask if they are interested
   - Suggest a meeting tonight or tomorrow
   - Mention the 20% discount for first meeting
   - Offer to send portfolio discreetly (photos of ladies)
   - IF YOU AGREE ON A MEETING: Remember the time and day

5. HANDLING OBJECTIONS:
   - "Not sure" → Emphasize discretion and professionalism. Offer to send information without obligation
   - "Too expensive" → Explain the quality of service and mention the first-time discount
   - "No time" → "I understand. We work 24/7, we can arrange a meeting whenever convenient for you"
   - "Married/In a relationship" → Assure them of complete discretion and confidentiality

IMPORTANT: 
- Be professional, polite, and respect personal boundaries
- NEVER be vulgar or rude
- Emphasize discretion and safety
- Respect refusal - if the client is not interested, thank them politely and end the call
- Speak naturally and confidently, without sounding like a salesperson """
        
        # Add main benefits if provided
        if main_benefits:
            system_text += f"Key benefits to emphasize: {main_benefits}. "
        
        # Add special offers if provided
        if special_offer:
            system_text += f"Current offers: {special_offer}. "
        
        # Add urgency based on setting
        if call_urgency == "high":
            system_text += "Create MAXIMUM URGENCY - emphasize time-sensitive offers and limited availability. "
        elif call_urgency == "medium":
            system_text += "Create moderate urgency with special offers and time-sensitive deals. "
        else:
            system_text += "Be persistent but not overly aggressive. Focus on building rapport and trust. "
        
        # Add objection handling strategy
        if objection_strategy == "understanding":
            system_text += "Handle objections with empathy and understanding. Listen to their concerns and address them thoughtfully. "
        elif objection_strategy == "educational":
            system_text += "Handle objections by providing educational information and facts to overcome doubts. "
        elif objection_strategy == "aggressive":
            system_text += "Handle objections persistently. Push back on concerns and maintain strong sales pressure. "
        
        system_text += f"Always maintain professionalism and use formal address ({formal_address}). Speak clearly, enthusiastically, and with confidence."
        
        # Add technical context about audio quality
        system_text += " You are speaking over a legacy telephone line. "
        system_text += "The audio quality may be low, muffled, or contain static. "
        system_text += "If you cannot clearly understand what the user said, DO NOT GUESS. "
        system_text += "Instead, politely ask them to repeat themselves or ask 'I'm sorry, the connection is bad, could you say that again?' "
        system_text += "Do not hallucinate answers based on static noise."
        
        # Add additional prompt if provided
        if additional_prompt:
            system_text += f" Additional instructions: {additional_prompt}"
    
    # Return a types.LiveConnectConfig object with the new structure
    live_cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity="START_SENSITIVITY_HIGH",
                end_of_speech_sensitivity="END_SENSITIVITY_HIGH",  # HIGH = aggressive end detection for lowest latency
                prefix_padding_ms=0,  # LATENCY FIX: No padding for minimum latency
                silence_duration_ms=50,  # AGGRESSIVE LATENCY: Reduced to 50ms for instant responses
            )
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)  # ✅ Use voice from CRM
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=system_text)]
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
    )
    
    # ✅ Try to enable server-side transcripts for both directions.
    # The SDK has 'input_audio_transcription' and 'output_audio_transcription' fields
    # which both use AudioTranscriptionConfig
    try:
        from google.genai import types as gtypes
        
        # Check if AudioTranscriptionConfig exists
        if hasattr(gtypes, "AudioTranscriptionConfig"):
            # Enable both input (user) and output (agent) transcription
            live_cfg.input_audio_transcription = gtypes.AudioTranscriptionConfig()
            live_cfg.output_audio_transcription = gtypes.AudioTranscriptionConfig()
            logger.info("✅ Enabled live audio transcription for both input (user) and output (agent)")
        else:
            logger.warning("⚠️ AudioTranscriptionConfig not found in SDK - live transcription not available")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not enable live transcripts: {e}. Continuing with AUDIO only.")
    
    return live_cfg

# Default voice config (Bulgarian) - will be overridden dynamically
DEFAULT_VOICE_CONFIG = create_voice_config(get_language_config('BG'))
MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

class AudioTranscriber:
    """Handles audio transcription using multiple methods with Windows-compatible fallbacks"""
    
    def __init__(self, model_size: str = "large"):
        """
        Initialize the transcriber with specified model size.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
                       'large' provides the best accuracy for all languages including Bulgarian
        """
        # Set transcription method and availability first
        self.transcription_method = TRANSCRIPTION_METHOD
        self.available = TRANSCRIPTION_AVAILABLE
        
        # Auto-select the best model size based on available method and system
        self.model_size = self._select_best_model_size(model_size)
        self.model = None
        self.model_loaded = False
        
        # Initialize Gemini client if using Gemini API method
        if self.transcription_method == "gemini_api":
            try:
                from google import genai
                self.gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
                logger.info("✅ Gemini API client initialized for fast transcription")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini client: {e}")
                self.available = False
        
        # Initialize OpenAI client if using API method
        if self.transcription_method == "openai_api":
            try:
                import openai
                import os
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.available = False
        
        if self.available:
            logger.info(f"🎤 AudioTranscriber initialized using {self.transcription_method} with model size: {self.model_size}")
            if self.model_size != model_size:
                logger.info(f"📈 Auto-selected model size '{self.model_size}' (requested: '{model_size}') for optimal Bulgarian transcription")
        else:
            logger.warning(f"⚠️ AudioTranscriber initialized but no transcription method available - transcription disabled")
    
    def _select_best_model_size(self, requested_size: str) -> str:
        """
        Automatically select the best model size based on transcription method and system capabilities.
        
        For Bulgarian and other non-English languages, larger models provide significantly better accuracy.
        """
        # If using Gemini API, model size doesn't matter (it's a cloud model)
        if self.transcription_method == "gemini_api":
            logger.info("🚀 Using Gemini API - cloud model with automatic optimization")
            return "large"  # Return large to indicate best quality, but doesn't affect actual API usage
        
        # If using OpenAI API, always use their best model (whisper-1 which is equivalent to large)
        if self.transcription_method == "openai_api":
            logger.info("🌐 Using OpenAI API - will use their best model (whisper-1)")
            return "large"  # This doesn't matter for API but keep consistent
        
        # For local transcription methods, prioritize accuracy over speed
        # Bulgarian and other Slavic languages benefit greatly from larger models
        model_priority = ["large", "medium", "small", "base", "tiny"]
        
        # Try to use the largest model available for best Bulgarian transcription
        if self.transcription_method in ["openai_whisper", "faster_whisper"]:
            # For Bulgarian transcription, we want the highest accuracy possible
            if requested_size == "large" or not requested_size:
                logger.info("🎯 Using 'large' model for optimal Bulgarian and multilingual transcription accuracy")
                return "large"
            else:
                # Still try to use a good model even if smaller was requested
                if requested_size in ["tiny", "base"]:
                    logger.info(f"📈 Upgrading from '{requested_size}' to 'medium' for better Bulgarian transcription")
                    return "medium"
                else:
                    return requested_size
        
        # Fallback to requested size if no specific optimization
        return requested_size
    
    def _load_model(self):
        """Lazy load the model when first needed (based on transcription method)"""
        if not self.available:
            raise RuntimeError("No transcription method is available - cannot load model")
            
        if not self.model_loaded:
            try:
                if self.transcription_method == "gemini_api":
                    # No model loading needed for Gemini API - client is already initialized
                    logger.info(f"✅ Gemini API client ready for fast transcription")
                    
                elif self.transcription_method == "faster_whisper":
                    logger.info(f"📥 Loading faster-whisper model '{self.model_size}'... (this may take a moment)")
                    from faster_whisper import WhisperModel
                    self.model = WhisperModel(self.model_size, device="cpu")
                    logger.info(f"✅ faster-whisper model '{self.model_size}' loaded successfully")
                    
                elif self.transcription_method == "openai_whisper":
                    logger.info(f"📥 Loading openai-whisper model '{self.model_size}'... (this may take a moment)")
                    import whisper
                    self.model = whisper.load_model(self.model_size)
                    logger.info(f"✅ openai-whisper model '{self.model_size}' loaded successfully")
                    
                elif self.transcription_method == "openai_api":
                    # No model loading needed for API - client is already initialized
                    logger.info(f"✅ OpenAI API client ready for transcription")
                    
                self.model_loaded = True
                
            except Exception as e:
                logger.error(f"❌ Failed to load {self.transcription_method} model: {e}")
                raise
    
    def transcribe_audio_file(self, audio_path: str, language: str = None) -> dict:
        """
        Transcribe audio file to text using available transcription method.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            language: Optional language code to hint the model (e.g., 'en', 'bg', 'ro')
        
        Returns:
            dict: Transcription result with 'text', 'language', 'segments' etc.
        """
        if not self.available:
            logger.warning(f"⚠️ Transcription requested but no method available for {audio_path}")
            return {
                "text": "",
                "error": "No transcription method available - transcription disabled",
                "success": False
            }
        
        try:
            # Check if audio file exists
            if not Path(audio_path).exists():
                logger.error(f"Audio file not found: {audio_path}")
                return {"text": "", "error": "File not found", "success": False}
            
            # Get file size for logging
            file_size = Path(audio_path).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"🎙️ Transcribing audio file using {self.transcription_method}: {audio_path} ({file_size:.2f} MB)")
            
            # Transcribe using the available method
            if self.transcription_method == "gemini_api":
                result = self._transcribe_with_gemini_api(audio_path, language)
            elif self.transcription_method == "openai_api":
                result = self._transcribe_with_openai_api(audio_path, language)
            else:
                # Load local model if not already loaded
                if not self.model_loaded:
                    self._load_model()
                
                if self.transcription_method == "faster_whisper":
                    result = self._transcribe_with_faster_whisper(audio_path, language)
                elif self.transcription_method == "openai_whisper":
                    result = self._transcribe_with_openai_whisper(audio_path, language)
                else:
                    raise ValueError(f"Unknown transcription method: {self.transcription_method}")
            
            # Log transcription results
            detected_language = result.get('language', 'unknown')
            text = result.get('text', '').strip()
            text_length = len(text)
            confidence = result.get('confidence')
            
            logger.info(f"✅ Transcription completed")
            logger.info(f"   Method: {self.transcription_method}")
            logger.info(f"   Detected language: {detected_language}")
            logger.info(f"   Text length: {text_length} characters")
            if confidence:
                logger.info(f"   Confidence: {confidence:.3f}")
            logger.info(f"   Text preview: {text[:100]}..." if text_length > 100 else f"   Full text: {text}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error transcribing audio file {audio_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def _transcribe_with_gemini_api(self, audio_path: str, language: str = None) -> dict:
        """Transcribe using Gemini API - fast and accurate transcription"""
        try:
            logger.info("🚀 Using Gemini API for fast transcription")
            if language:
                lang_name = {
                    'bg': 'Bulgarian', 'en': 'English', 'ro': 'Romanian', 
                    'el': 'Greek', 'de': 'German', 'fr': 'French', 
                    'es': 'Spanish', 'it': 'Italian', 'ru': 'Russian'
                }.get(language, language)
                logger.info(f"🎯 Language hint: {lang_name}")
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Prepare the audio file for Gemini
            # Gemini accepts various audio formats including WAV
            audio_file_part = {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(audio_data).decode('utf-8')
                }
            }
            
            # Create the transcription prompt
            if language:
                lang_name = {
                    'bg': 'Bulgarian', 'en': 'English', 'ro': 'Romanian',
                    'el': 'Greek', 'de': 'German', 'fr': 'French',
                    'es': 'Spanish', 'it': 'Italian', 'ru': 'Russian'
                }.get(language, language)
                prompt = f"Please transcribe this audio in {lang_name}. Provide only the transcription text, nothing else."
            else:
                prompt = "Please transcribe this audio. Provide only the transcription text, nothing else."
            
            # Use Gemini 2.5 Flash model (paid, supports audio transcription)
            model_id = "gemini-2.5-flash"
            
            logger.info(f"📤 Sending audio to Gemini model: {model_id}")
            
            # Generate content with audio
            response = self.gemini_client.models.generate_content(
                model=model_id,
                contents=[
                    prompt,
                    audio_file_part
                ]
            )
            
            # Extract transcription text
            transcript_text = ""
            if hasattr(response, 'text'):
                transcript_text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text'):
                                transcript_text += part.text
            
            transcript_text = transcript_text.strip()
            
            # Try to detect language from result if not provided
            detected_language = language or 'unknown'
            
            logger.info(f"✅ Gemini API transcription completed: {len(transcript_text)} chars")
            
            return {
                "text": transcript_text,
                "language": detected_language,
                "segments": [],  # Gemini doesn't provide segments by default
                "confidence": None,  # Gemini doesn't provide confidence scores
                "success": True,
                "method": "gemini_api"
            }
        except Exception as e:
            logger.error(f"❌ Gemini API transcription failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _transcribe_with_openai_api(self, audio_path: str, language: str = None) -> dict:
        """Transcribe using OpenAI's cloud API with optimal settings"""
        try:
            logger.info("🌐 Using OpenAI API (whisper-1 model) for transcription")
            if language:
                logger.info(f"🎯 Language hint: {language}")
                if language in ['bg', 'Bulgarian']:
                    logger.info("🇧🇬 Bulgarian language detected - using optimal API settings")
            
            with open(audio_path, 'rb') as audio_file:
                # Use verbose_json for detailed results including timestamps
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",  # OpenAI's best Whisper model
                    file=audio_file,
                    language=language if language and len(language) == 2 else None,  # Only 2-letter codes for API
                    response_format="verbose_json",  # Get detailed results with segments
                    temperature=0.0,  # Most deterministic results
                    # Note: OpenAI API doesn't support all the local model parameters
                )
            
            # Extract segments if available
            segments = []
            if hasattr(transcript, 'segments') and transcript.segments:
                for segment in transcript.segments:
                    segments.append({
                        "start": getattr(segment, 'start', 0),
                        "end": getattr(segment, 'end', 0),
                        "text": getattr(segment, 'text', ''),
                        "confidence": None  # API doesn't provide per-segment confidence
                    })
            
            result_text = transcript.text.strip()
            detected_language = getattr(transcript, 'language', language or 'unknown')
            
            logger.info(f"✅ OpenAI API transcription completed: {len(result_text)} chars, language: {detected_language}")
            
            return {
                "text": result_text,
                "language": detected_language,
                "segments": segments,
                "confidence": None,  # OpenAI API doesn't provide confidence scores
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ OpenAI API transcription failed: {e}")
            raise
    
    def _transcribe_with_faster_whisper(self, audio_path: str, language: str = None) -> dict:
        """Transcribe using faster-whisper (local) with optimal settings"""
        try:
            # Prepare transcription options with optimal settings for accuracy
            options = {
                # Use beam search for better accuracy
                'beam_size': 5,
                # More conservative thresholds for better accuracy
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                # Enable word-level timestamps
                'word_timestamps': True,
                # Lower temperature for more consistent results
                'temperature': 0.0,
                # Better handling of longer audio
                'condition_on_previous_text': True,
            }
            
            if language:
                options['language'] = language
                logger.info(f"🎯 Transcribing with language hint: {language}")
                
                # For Bulgarian and other Slavic languages, use specific optimizations
                if language in ['bg', 'Bulgarian']:
                    logger.info("🇧🇬 Applying Bulgarian transcription optimizations")
                    # Even more conservative settings for Bulgarian accuracy
                    options['beam_size'] = 10  # More thorough search for Bulgarian
                    options['temperature'] = 0.0
                    options['compression_ratio_threshold'] = 2.2  # Stricter threshold
                    options['logprob_threshold'] = -0.8  # Higher confidence requirement
            
            logger.info(f"🎤 Transcribing with faster-whisper options: {options}")
            segments, info = self.model.transcribe(audio_path, **options)
            
            # Collect all segments into text
            text_segments = []
            all_segments = []
            total_confidence = 0
            segment_count = 0
            
            for segment in segments:
                text_segments.append(segment.text)
                all_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": getattr(segment, 'avg_logprob', 0)
                })
                if hasattr(segment, 'avg_logprob'):
                    total_confidence += segment.avg_logprob
                    segment_count += 1
            
            full_text = ' '.join(text_segments).strip()
            avg_confidence = total_confidence / segment_count if segment_count > 0 else None
            
            logger.info(f"✅ Transcribed {len(text_segments)} segments with avg confidence: {avg_confidence:.3f}" if avg_confidence else "✅ Transcription completed")
            
            return {
                "text": full_text,
                "language": info.language,
                "segments": all_segments,
                "confidence": avg_confidence,
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ faster-whisper transcription failed: {e}")
            raise
    
    def _transcribe_with_openai_whisper(self, audio_path: str, language: str = None) -> dict:
        """Transcribe using original openai-whisper (local) with optimal settings"""
        try:
            # Prepare transcription options with optimal settings for accuracy
            options = {
                # Use beam search for better accuracy (especially important for Bulgarian)
                'beam_size': 5,
                # Lower temperature for more consistent results
                'temperature': 0.0,
                # Enable word-level timestamps for detailed analysis
                'word_timestamps': True,
                # Use more aggressive decoding for non-English languages
                'condition_on_previous_text': True,
            }
            
            if language:
                options['language'] = language
                logger.info(f"🎯 Transcribing with language hint: {language}")
                
                # For Bulgarian and other Slavic languages, use specific optimizations
                if language in ['bg', 'Bulgarian']:
                    logger.info("🇧🇬 Applying Bulgarian transcription optimizations")
                    options['temperature'] = 0.0  # More deterministic for Bulgarian
                    options['compression_ratio_threshold'] = 2.4
                    options['logprob_threshold'] = -1.0
                    options['no_speech_threshold'] = 0.6
            
            logger.info(f"🎤 Transcribing with options: {options}")
            
            # Try transcription with word timestamps first
            try:
                result = self.model.transcribe(audio_path, **options)
                return {
                    "text": result.get('text', '').strip(),
                    "language": result.get('language', 'unknown'),
                    "segments": result.get('segments', []),
                    "confidence": getattr(result, 'avg_logprob', None),
                    "success": True
                }
            except Exception as timestamp_error:
                # If word timestamps fail (common with mixed audio), retry without them
                logger.warning(f"⚠️ Word timestamps failed, retrying without: {timestamp_error}")
                
                # Fallback options without word timestamps
                fallback_options = options.copy()
                fallback_options['word_timestamps'] = False
                
                logger.info(f"🔄 Retrying transcription without word timestamps...")
                result = self.model.transcribe(audio_path, **fallback_options)
                
                return {
                    "text": result.get('text', '').strip(),
                    "language": result.get('language', 'unknown'),
                    "segments": result.get('segments', []),
                    "confidence": getattr(result, 'avg_logprob', None),
                    "success": True,
                    "fallback_used": "no_word_timestamps"
                }
            
        except Exception as e:
            logger.error(f"❌ openai-whisper transcription failed: {e}")
            raise
    
    def transcribe_call_recordings(self, session_dir: Path, language_hint: str = None) -> dict:
        """
        Transcribe all audio files for a call session concurrently.
        
        Args:
            session_dir: Path to session directory containing WAV files
            language_hint: Optional language code to hint the model
        
        Returns:
            dict: Transcription results for incoming, outgoing, and mixed audio
        """
        # Use asyncio to run concurrent transcription
        import asyncio
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, create a new task
            return asyncio.run_coroutine_threadsafe(
                self._transcribe_call_recordings_concurrent(session_dir, language_hint),
                loop
            ).result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(self._transcribe_call_recordings_concurrent(session_dir, language_hint))
    
    async def _transcribe_call_recordings_concurrent(self, session_dir: Path, language_hint: str = None) -> dict:
        """
        Internal async method to handle concurrent transcription.
        """
        transcripts = {}
        
        if not self.available:
            logger.warning(f"⚠️ Transcription requested for session {session_dir.name} but Whisper not available")
            # Return empty results for all audio types so the system doesn't break
            for audio_type in ['incoming', 'outgoing', 'mixed']:
                transcripts[audio_type] = {
                    "text": "",
                    "error": "Whisper library not available - transcription disabled",
                    "success": False
                }
            return transcripts
        
        # Find audio files in session directory
        audio_files = {
            'incoming': None,
            'outgoing': None,
            'mixed': None
        }
        
        # Look for audio files
        for file_path in session_dir.glob("*.wav"):
            filename = file_path.name.lower()
            if 'incoming' in filename:
                audio_files['incoming'] = file_path
            elif 'outgoing' in filename:
                audio_files['outgoing'] = file_path
            elif 'mixed' in filename:
                audio_files['mixed'] = file_path
        
        # Prepare tasks for concurrent execution
        transcription_tasks = []
        audio_types_to_process = []
        
        for audio_type, audio_path in audio_files.items():
            if audio_path and audio_path.exists():
                audio_types_to_process.append(audio_type)
                transcription_tasks.append(
                    self._transcribe_single_audio_async(audio_type, audio_path, session_dir, language_hint)
                )
            else:
                logger.warning(f"⚠️ No {audio_type} audio file found in {session_dir}")
                # Create empty result for missing files
                transcripts[audio_type] = {
                    "text": "",
                    "error": f"No {audio_type} audio file found",
                    "success": False
                }
        
        if transcription_tasks:
            logger.info(f"🚀 Starting concurrent transcription of {len(transcription_tasks)} audio files...")
            start_time = time.time()
            
            # Run all transcription tasks concurrently
            results = await asyncio.gather(*transcription_tasks, return_exceptions=True)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Concurrent transcription completed in {elapsed_time:.2f} seconds")
            
            # Process results
            for i, (audio_type, result) in enumerate(zip(audio_types_to_process, results)):
                if isinstance(result, Exception):
                    logger.error(f"❌ Transcription failed for {audio_type}: {result}")
                    transcripts[audio_type] = {
                        "text": "",
                        "error": f"Transcription failed: {str(result)}",
                        "success": False
                    }
                else:
                    transcripts[audio_type] = result
        
        return transcripts
    
    async def _transcribe_single_audio_async(self, audio_type: str, audio_path: Path, session_dir: Path, language_hint: str = None) -> dict:
        """
        Transcribe a single audio file asynchronously.
        """
        import asyncio
        import concurrent.futures
        
        logger.info(f"🎙️ Starting transcription of {audio_type} audio...")
        
        # Use ThreadPoolExecutor to run the sync transcription in a separate thread
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                # Run the synchronous transcription in a thread
                result = await loop.run_in_executor(
                    executor,
                    self.transcribe_audio_file,
                    str(audio_path),
                    language_hint
                )
                
                # Save transcript to text file (even if transcription failed)
                transcript_path = session_dir / f"{audio_type}_transcript.txt"
                await self._save_transcript_file_async(transcript_path, audio_path, result, audio_type)
                
                logger.info(f"✅ Completed transcription of {audio_type} audio ({len(result.get('text', ''))} characters)")
                result['transcript_file'] = str(transcript_path.name)
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Error transcribing {audio_type} audio: {e}")
                return {
                    "text": "",
                    "error": f"Transcription failed: {str(e)}",
                    "success": False
                }
    
    async def _save_transcript_file_async(self, transcript_path: Path, audio_path: Path, result: dict, audio_type: str):
        """
        Save transcript file asynchronously.
        """
        import asyncio
        
        def _write_transcript():
            try:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {audio_type.title()} Audio Transcript\n")
                    f.write(f"# File: {audio_path.name}\n")
                    f.write(f"# Language: {result.get('language', 'unknown')}\n")
                    f.write(f"# Transcribed: {datetime.now(timezone.utc).isoformat()}\n")
                    
                    if result.get('success', False):
                        f.write(f"\n{result.get('text', '')}")
                        
                        # Add detailed segments if available
                        segments = result.get('segments', [])
                        if segments:
                            f.write("\n\n# Detailed Segments (with timestamps)\n")
                            for segment in segments:
                                start = segment.get('start', 0)
                                end = segment.get('end', 0)
                                text = segment.get('text', '').strip()
                                f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
                    else:
                        f.write(f"\n# Transcription failed: {result.get('error', 'Unknown error')}\n")
                
                logger.info(f"💾 Saved transcript: {transcript_path}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error saving transcript file: {e}")
                return False
        
        # Run the file writing in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _write_transcript)

# Global transcriber instance (lazy-loaded) - using large model for best accuracy
audio_transcriber = AudioTranscriber(model_size="large")

class CallRecorder:
    """Records call audio to WAV files - separate files for incoming and outgoing audio
    
    This class handles local audio recording for call sessions by:
    - Recording incoming audio from caller
    - Recording outgoing audio (agent/AI responses)
    - Creating mixed/combined recording
    - Saving session metadata
    """
    
    def __init__(self, session_id: str, caller_id: str, called_number: str):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        
        # Create session directory
        self.session_dir = Path(f"sessions/{session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # WAV file paths
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.incoming_wav_path = self.session_dir / f"incoming_{timestamp}.wav"
        self.outgoing_wav_path = self.session_dir / f"outgoing_{timestamp}.wav"
        self.mixed_wav_path = self.session_dir / f"mixed_{timestamp}.wav"
        
        # WAV file handles
        self.incoming_wav = None
        self.outgoing_wav = None
        
        # Audio buffers for mixing
        self.incoming_buffer = []
        self.outgoing_buffer = []
        
        # Recording state
        self.recording = True
        
        # Initialize WAV files
        self._initialize_wav_files()
        
        logger.info(f"🎙️ Recording initialized for session {session_id}")
        logger.info(f"   Incoming: {self.incoming_wav_path}")
        logger.info(f"   Outgoing: {self.outgoing_wav_path}")
    
    def _initialize_wav_files(self):
        """Initialize WAV files for recording"""
        try:
            # Incoming audio (from caller) - 8kHz, 16-bit, mono
            self.incoming_wav = wave.open(str(self.incoming_wav_path), 'wb')
            self.incoming_wav.setnchannels(1)  # Mono
            self.incoming_wav.setsampwidth(2)  # 16-bit
            self.incoming_wav.setframerate(8000)  # 8kHz
            
            # Outgoing audio (to caller) - 8kHz, 16-bit, mono  
            self.outgoing_wav = wave.open(str(self.outgoing_wav_path), 'wb')
            self.outgoing_wav.setnchannels(1)  # Mono
            self.outgoing_wav.setsampwidth(2)  # 16-bit
            self.outgoing_wav.setframerate(8000)  # 8kHz
            
        except Exception as e:
            logger.error(f"Failed to initialize WAV files: {e}")
            self.recording = False
    
    def record_incoming_audio(self, pcm_data: bytes):
        """Record incoming audio (from caller)"""
        if not self.recording or not self.incoming_wav:
            return
            
        try:
            self.incoming_wav.writeframes(pcm_data)
            # Store in buffer for mixing
            self.incoming_buffer.append(pcm_data)
            
            # Keep buffer manageable (last 10 seconds at 8kHz)
            max_buffer_size = 10 * 8000 * 2  # 10 seconds of 16-bit audio
            current_size = sum(len(data) for data in self.incoming_buffer)
            while current_size > max_buffer_size and self.incoming_buffer:
                removed = self.incoming_buffer.pop(0)
                current_size -= len(removed)
                
        except Exception as e:
            logger.error(f"Error recording incoming audio: {e}")
    
    def record_outgoing_audio(self, pcm_data: bytes):
        """Record outgoing audio (to caller)"""
        if not self.recording or not self.outgoing_wav:
            return
            
        try:
            self.outgoing_wav.writeframes(pcm_data)
            # Store in buffer for mixing
            self.outgoing_buffer.append(pcm_data)
            
            # Keep buffer manageable (last 10 seconds at 8kHz)
            max_buffer_size = 10 * 8000 * 2  # 10 seconds of 16-bit audio
            current_size = sum(len(data) for data in self.outgoing_buffer)
            while current_size > max_buffer_size and self.outgoing_buffer:
                removed = self.outgoing_buffer.pop(0)
                current_size -= len(removed)
                
        except Exception as e:
            logger.error(f"Error recording outgoing audio: {e}")
    
    def create_mixed_recording(self):
        """Create a mixed recording with both incoming and outgoing audio"""
        if not self.recording:
            return
            
        try:
            logger.info("🎵 Creating mixed audio recording...")
            
            # Read both WAV files
            incoming_audio = b""
            outgoing_audio = b""
            
            # Read incoming audio
            if self.incoming_wav_path.exists():
                with wave.open(str(self.incoming_wav_path), 'rb') as wav:
                    incoming_audio = wav.readframes(wav.getnframes())
            
            # Read outgoing audio  
            if self.outgoing_wav_path.exists():
                with wave.open(str(self.outgoing_wav_path), 'rb') as wav:
                    outgoing_audio = wav.readframes(wav.getnframes())
            
            # Convert to numpy arrays for mixing
            # Mix audio using audioop
            import audioop
            
            # Ensure we have bytes
            incoming_audio = incoming_audio or b""
            outgoing_audio = outgoing_audio or b""
            
            incoming_len = len(incoming_audio)
            outgoing_len = len(outgoing_audio)
            max_len = max(incoming_len, outgoing_len)
            
            # Pad with silence
            if incoming_len < max_len:
                incoming_audio += b'\x00' * (max_len - incoming_len)
            if outgoing_len < max_len:
                outgoing_audio += b'\x00' * (max_len - outgoing_len)
            
            if max_len > 0:
                # Average to prevent clipping: (a/2 + b/2)
                # This works well for simple mixing
                inc_scaled = audioop.mul(incoming_audio, 2, 0.5)
                out_scaled = audioop.mul(outgoing_audio, 2, 0.5)
                mixed_data = audioop.add(inc_scaled, out_scaled, 2)
                
                # Save mixed audio
                with wave.open(str(self.mixed_wav_path), 'wb') as mixed_wav:
                    mixed_wav.setnchannels(1)  # Mono
                    mixed_wav.setsampwidth(2)  # 16-bit
                    mixed_wav.setframerate(8000)  # 8kHz
                    mixed_wav.writeframes(mixed_data)
                
                logger.info(f"✅ Mixed recording saved: {self.mixed_wav_path}")
            else:
                logger.warning("No audio data to mix")
                
        except Exception as e:
            logger.error(f"Error creating mixed recording: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def stop_recording(self):
        """Stop recording and close WAV files"""
        if not self.recording:
            return
        
        # Check if Python is shutting down before doing file operations
        import sys
        if sys.meta_path is None:
            logger.debug("Python shutting down, skipping recording cleanup")
            return
            
        try:
            logger.info(f"🛑 Stopping recording for session {self.session_id}")
            
            # Close WAV files
            if self.incoming_wav:
                try:
                    self.incoming_wav.close()
                except:
                    pass
                self.incoming_wav = None
                
            if self.outgoing_wav:
                try:
                    self.outgoing_wav.close()
                except:
                    pass
                self.outgoing_wav = None
            
            # Create mixed recording (only if not shutting down)
            try:
                self.create_mixed_recording()
            except Exception as e:
                # Don't log during shutdown as logging may not be available
                if sys.meta_path is not None:
                    logger.warning(f"Could not create mixed recording: {e}")
            
            # Create session info file
            try:
                self._save_session_info()
            except Exception as e:
                if sys.meta_path is not None:
                    logger.warning(f"Could not save session info: {e}")
            
            self.recording = False
            
            # Log file sizes and locations
            if self.incoming_wav_path.exists():
                size_mb = self.incoming_wav_path.stat().st_size / (1024 * 1024)
                logger.info(f"📁 Incoming audio: {self.incoming_wav_path} ({size_mb:.2f} MB)")
                
            if self.outgoing_wav_path.exists():
                size_mb = self.outgoing_wav_path.stat().st_size / (1024 * 1024)
                logger.info(f"📁 Outgoing audio: {self.outgoing_wav_path} ({size_mb:.2f} MB)")
                
            if self.mixed_wav_path.exists():
                size_mb = self.mixed_wav_path.stat().st_size / (1024 * 1024)
                logger.info(f"📁 Mixed audio: {self.mixed_wav_path} ({size_mb:.2f} MB)")
            
            # Automatic transcription disabled - can be triggered manually through CRM
            logger.info(f"📼 Call recording completed for session {self.session_id}")
            if TRANSCRIPTION_AVAILABLE:
                logger.info(f"🎤 Transcription available - can be triggered manually through CRM interface")
            else:
                logger.info(f"⚠️ Transcription service not available")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Note: Transcription is now handled by the standalone transcribe_audio.py script
    # It's triggered manually through the CRM interface or via the API endpoint
    # The script handles updating session_info.json automatically
    # See _background_transcribe_session() for the API implementation
    
    def _save_session_info(self):
        """Save session information to a JSON file"""
        try:
            call_duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            session_info = {
                "session_id": self.session_id,
                "caller_id": self.caller_id,
                "called_number": self.called_number,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": call_duration,
                "files": {
                    "incoming_audio": str(self.incoming_wav_path.name),
                    "outgoing_audio": str(self.outgoing_wav_path.name),
                    "mixed_audio": str(self.mixed_wav_path.name)
                }
            }
            
            info_path = self.session_dir / "session_info.json"
            with open(info_path, 'w') as f:
                json.dump(session_info, f, indent=2)
                
            logger.info(f"📄 Session info saved: {info_path}")
            
            # Save to MongoDB
            try:
                from session_mongodb_helper import save_session_info_to_mongodb
                save_session_info_to_mongodb(self.session_id, self.session_dir)
            except Exception as mongo_error:
                logger.warning(f"⚠️  Could not save session info to MongoDB: {mongo_error}")
            
        except Exception as e:
            logger.error(f"Error saving session info: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.recording:
            self.stop_recording()

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
            
            # --- FIX START: OPTIMIZED SOCKET BUFFER ---
            # Low latency settings: Keep Receive buffer large for safety, 
            # but drastically REDUCE Send buffer to prevent "ghost audio" after interruption.
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)  # 64KB for low latency
                # Set Send buffer to ~1 second max (8KB/s * 1s = ~8192 bytes). 
                # 16384 gives us a safe margin without buffering seconds of audio.
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024)
            except Exception as e:
                logger.warning(f"Could not increase socket buffer: {e}")
            # --- FIX END ---

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
                    
                    # Find session for this RTP stream
                    # Priority 1: Match by SSRC (most reliable for concurrent calls)
                    # Priority 2: Match by port (different calls use different ports)
                    # Priority 3: Match by IP only for UNINITIALIZED sessions (initialization)
                    session_found = False
                    matched_session = None
                    uninitialized_session = None
                    
                    for session_id, rtp_session in self.sessions.items():
                        # First, try to match by SSRC if we've seen it before
                        if hasattr(rtp_session, 'remote_ssrc') and rtp_session.remote_ssrc == ssrc:
                            matched_session = rtp_session
                            session_found = True
                            logger.debug(f"✅ Matched RTP packet by SSRC {ssrc} to session {session_id}")
                            break
                        
                        # Second, try to match by IP and port (works for most cases)
                        if rtp_session.remote_addr == addr:
                            matched_session = rtp_session
                            # Store OR Update the SSRC (in case it changes mid-call)
                            if not hasattr(rtp_session, 'remote_ssrc') or rtp_session.remote_ssrc != ssrc:
                                old_ssrc = getattr(rtp_session, 'remote_ssrc', 'None')
                                rtp_session.remote_ssrc = ssrc
                                logger.info(f"🔒 Updated RTP session {session_id} SSRC from {old_ssrc} to {ssrc} (Address Match)")
                            session_found = True
                            break
                        
                        # Track uninitialized sessions (port still 5004) for later
                        if not hasattr(rtp_session, 'rtp_initialized') and rtp_session.remote_addr[1] == 5004:
                            # Only consider if not already claimed by another SSRC
                            if not hasattr(rtp_session, 'remote_ssrc'):
                                if uninitialized_session is None:
                                    uninitialized_session = (session_id, rtp_session)
                    
                    # If no match found, try to initialize the first uninitialized session
                    if not session_found and uninitialized_session is not None:
                        session_id, rtp_session = uninitialized_session
                        logger.info(f"📍 Initializing RTP session {session_id} with actual port {addr[1]} and SSRC {ssrc}")
                        rtp_session.remote_addr = addr
                        rtp_session.remote_ssrc = ssrc
                        rtp_session.rtp_initialized = True
                        matched_session = rtp_session
                        session_found = True
                    
                    if matched_session:
                        matched_session.process_incoming_audio(audio_payload, payload_type, timestamp)
                    elif not session_found:
                        # No session found for this address - log only once per address
                        if not hasattr(self, '_unknown_addresses'):
                            self._unknown_addresses = set()
                        if addr not in self._unknown_addresses:
                            logger.warning(f"⚠️ Received RTP audio from unknown address {addr} (SSRC: {ssrc})")
                            self._unknown_addresses.add(addr)
                
            except Exception as e:
                logger.error(f"Error in RTP listener: {e}")
    
    def create_session(self, session_id: str, remote_addr, voice_session, call_recorder=None):
        """Create RTP session for a call"""
        rtp_session = RTPSession(session_id, remote_addr, self.socket, voice_session, call_recorder)
        self.sessions[session_id] = rtp_session
        logger.info(f"🎵 Created RTP session {session_id} for {remote_addr}")
        return rtp_session
    
    def remove_session(self, session_id: str):
        """Remove RTP session and cleanup its resources"""
        if session_id in self.sessions:
            rtp_session = self.sessions[session_id]
            
            # Call cleanup method to stop all threads
            if hasattr(rtp_session, 'cleanup_threads'):
                try:
                    rtp_session.cleanup_threads()
                    logger.info(f"✅ RTP session {session_id} threads cleaned up")
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning up RTP session threads: {e}")
            
            del self.sessions[session_id]
            logger.info(f"🎵 Removed RTP session {session_id}")
        else:
            logger.debug(f"RTP session {session_id} already removed")
    
    def stop(self):
        """Stop RTP server"""
        self.running = False
        if self.socket:
            self.socket.close()

class RTPSession:
    """
    Individual RTP session for a call.
    
    PROFESSIONAL COLD CALLING ENHANCEMENTS:
    - Explicit conversation state machine
    - Answering Machine Detection (AMD)
    - Adaptive silence detection
    - Debounced end-of-turn detection
    - Better turn-taking with state awareness
    """
    
    def __init__(self, session_id: str, remote_addr, rtp_socket, voice_session, call_recorder=None):
        self.session_id = session_id
        self.remote_addr = remote_addr
        self.rtp_socket = rtp_socket
        self.voice_session = voice_session
        self.call_recorder = call_recorder  # Add call recorder
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = hash(session_id) & 0xFFFFFFFF
        
        # =================================================================
        # PROFESSIONAL COLD CALLING: Conversation State Machine
        # =================================================================
        self.conversation_state = ConversationState.INITIALIZING
        self._state_change_time = time.time()
        self._state_history = []  # Track state changes for debugging
        logger.info(f"🎯 Conversation state machine initialized: {self.conversation_state.name}")
        
        # =================================================================
        # PROFESSIONAL COLD CALLING: Answering Machine Detection
        # =================================================================
        self.amd = AnsweringMachineDetector(detection_window_sec=5.0)
        self.amd_result = None  # None=detecting, True=AMD, False=Human
        self._amd_action_taken = False  # Track if we acted on AMD result
        
        # =================================================================
        # PROFESSIONAL COLD CALLING: Call Quality Tracking
        # =================================================================
        self.quality_tracker = CallQualityTracker(session_id)
        
        # =================================================================
        # LATENCY TRACKING: Pipeline Performance Monitoring
        # =================================================================
        self.latency_tracker = LatencyTracker(session_id)
        
        # =================================================================
        # PROFESSIONAL COLD CALLING: Adaptive Silence Detection
        # =================================================================
        self.adaptive_silence = AdaptiveSilenceDetector(
            initial_threshold=5500.0,  # Increased for GSM noise rejection
            min_threshold=2500.0,
            max_threshold=12000.0,
            adaptation_rate=0.03  # Slower adaptation for stability
        )
        
        # =================================================================
        # PROFESSIONAL COLD CALLING: Debounced End-of-Turn Detection
        # =================================================================
        self.eot_detector = DebouncedEndOfTurnDetector(
            silence_threshold_sec=0.80,  # 800ms - allows natural pauses while thinking
            confirmation_frames=3,        # Require 3 consecutive silence frames
            cooldown_sec=0.5              # 500ms cooldown between triggers
        )
        
        # Audio processing - use queue for thread-safe asyncio communication
        self.input_processing = False  # For incoming audio processing
        self.audio_input_queue = queue.Queue()  # Queue audio for the asyncio thread
        self.audio_buffer = b""  # Small buffer for packet assembly only
        self.buffer_lock = threading.Lock()
        self.asyncio_loop = None  # Will hold the dedicated event loop
        self.asyncio_thread = None  # Dedicated thread for asyncio
        self.asyncio_thread_started = False  # Flag to prevent duplicate thread starts
        
        # Output audio queue for paced delivery
        self.output_queue = queue.Queue()
        self.output_thread = None
        self.output_processing = True  # For outgoing audio processing
        
        # Packet tracking for RTP activity detection
        self.packets_received = 0  # Count of received RTP packets
        
        # Audio level tracking for AGC
        self.audio_level_history = []
        self.target_audio_level = -20  # dBFS
        
        # Crossfade buffer for smooth transitions
        self.last_audio_tail = b""  # Last 10ms of previous chunk
        
        # Jitter buffer for smooth playback - ULTRA LOW LATENCY configuration
        # Start playing almost immediately upon receiving the first packet
        self.jitter_buffer = b""
        self.playback_started = False
        self.jitter_buffer_threshold = 320   # LATENCY FIX: 20ms at 8kHz 16-bit (reduced from 1600/100ms)
        self.jitter_buffer_min = 160         # LATENCY FIX: 10ms minimum (reduced from 800/50ms)
        self.jitter_buffer_max = 3200        # LATENCY FIX: 200ms max buffer (reduced from 4800/300ms)

        # Initialize professional audio preprocessor
        # 4-stage pipeline: Bandpass → Spectral Noise Suppression → AGC → Soft Limiter
        self.audio_preprocessor = AudioPreprocessor(sample_rate=8000)
        
        # Voice Activity Detection (Client-Side WebRTC VAD) - ENHANCED
        self.vad = VoiceActivityDetector(sample_rate=8000, aggressiveness=3)
        self.last_voice_activity_time = time.time()
        # CHANGE: Set to 1 hour (effectively disabled) to prevent local VAD from killing the call
        # The local VAD was too aggressive and would think the user was silent even when speaking
        self.voice_timeout_seconds = 3600.0  # 1 hour timeout (effectively disabled)
        self.timeout_monitor_thread = None
        self.timeout_monitoring = True
        self._cleanup_lock = threading.Lock()  # Prevent concurrent cleanup
        self.is_interrupted = False  # Barge-in interruption flag
        self._cleanup_done = False  # Track if cleanup has been performed
        
        # Barge-in VAD thread - runs independently to detect user speech and interrupt playback
        self.vad_queue = queue.Queue(maxsize=100)  # Queue for VAD audio samples
        self.vad_thread = None
        self.vad_thread_running = False
        self._user_speaking = False  # Track if user is currently speaking
        self._user_speech_start_time = 0  # Track when user started speaking
        self._barge_in_threshold_ms = 100  # Require 100ms of speech before interrupting
        logger.info(f"⏱️ Voice activity timeout monitoring enabled: {self.voice_timeout_seconds}s (user turn only)")
        
        # Goodbye detection for graceful hangup
        self.goodbye = GoodbyeDetector(grace_ms=1200)
        self._bye_sent = False  # Track if we already sent BYE
        logger.info("👋 Goodbye detection enabled for graceful hangup")
        
        # Full-duplex audio state management (for monitoring only - not blocking)
        self.assistant_speaking = False  # Track when assistant is speaking
        self.assistant_stop_time = 0  # Track when assistant stopped speaking
        self.echo_suppression_delay = 0.15  # 150ms reference time (used for logging only, not blocking)
        self.audio_output_active = False  # Track if we're actively outputting audio
        self.last_output_time = 0  # Track last time we sent output audio
        logger.info("🎙️ Full-duplex mode enabled - user audio always sent to Gemini for instant interruption")
        
        # Start output processing thread immediately for greeting playback
        self.output_thread = threading.Thread(target=self._process_output_queue, daemon=True)
        self.output_thread.start()
        logger.info(f"🎵 Started output queue processing for session {session_id}")
        logger.info(f"🎵 RTP session initialized: output_processing={self.output_processing}, remote_addr={remote_addr}")
        
        # Start timeout monitoring thread
        self.timeout_monitor_thread = threading.Thread(target=self._monitor_voice_timeout, daemon=True)
        self.timeout_monitor_thread.start()
        logger.info(f"⏱️ Started voice timeout monitoring for session {session_id}")
        
        # DISABLED: Local barge-in VAD thread - Gemini handles interruption detection server-side
        # This was causing issues with turn-taking. We now rely entirely on Gemini's native
        # voice activity detection and interruption handling.
        self.vad_thread_running = False
        self.vad_thread = None
        logger.info(f"🎤 Local VAD barge-in DISABLED - relying on Gemini server-side interruption detection")
        
        # End-of-turn detection state - signals Gemini when user stops speaking
        # NOTE: Now using DebouncedEndOfTurnDetector for professional turn-taking
        self._silence_start_time = None  # When silence started (legacy, kept for compatibility)
        self._end_of_turn_sent = False  # Whether we've sent end_of_turn for current silence period
        
        # --- PROFESSIONAL COLD CALLING: Improved thresholds ---
        # ⚡ LATENCY FIX: Increased silence threshold for natural Bulgarian speech patterns
        # 400ms was too short - users often pause mid-sentence, causing premature EOT
        # 700ms allows for natural pauses while still feeling responsive
        self._silence_threshold_sec = 0.35  # AGGRESSIVE: 350ms silence for faster turn-taking
        self._last_eot_time = 0  # Track last EOT time for cooldown
        
        # Higher threshold for GSM noise rejection (adaptive system will tune this)
        self._speech_energy_threshold = 5500  # PROFESSIONAL: Increased from 4200 for better noise rejection
        # --- END PROFESSIONAL FIX ---
        
        self._last_speech_time = time.time()  # Last time speech was detected
        logger.info(f"📍 PROFESSIONAL COLD CALLING: End-of-turn with {self._silence_threshold_sec}s silence, 2.5s EOT cooldown, adaptive threshold={self._speech_energy_threshold}")
        
        # Call answered flag - set to True when SIP 200 OK is received
        # Used to delay Gemini greeting until user actually picks up
        self.call_answered = False
        
        # --- GREETING PROTECTION ---
        # When True, blocks user audio from being sent to Gemini
        # This prevents the user's "Hello?" from interrupting the AI greeting
        self.greeting_protection_active = False
        
        # Keep processing flag for backward compatibility with other parts of the code
        self.processing = self.output_processing
    
    # =================================================================
    # PROFESSIONAL COLD CALLING: State Machine Methods
    # =================================================================
    
    def set_conversation_state(self, new_state: ConversationState, reason: str = ""):
        """
        Transition to a new conversation state with logging.
        
        Args:
            new_state: The new ConversationState to transition to
            reason: Optional reason for the state change
        """
        old_state = self.conversation_state
        if old_state == new_state:
            return  # No change
        
        self.conversation_state = new_state
        self._state_change_time = time.time()
        self._state_history.append((time.time(), old_state, new_state, reason))
        
        # Keep history manageable
        if len(self._state_history) > 50:
            self._state_history.pop(0)
        
        logger.info(f"🔄 State: {old_state.name} → {new_state.name}" + (f" ({reason})" if reason else ""))
        
        # Reset EOT detector on state change
        if new_state in (ConversationState.LISTENING, ConversationState.USER_SPEAKING):
            self.eot_detector.reset_for_new_turn()
    
    def get_state_duration(self) -> float:
        """Get how long we've been in the current state (seconds)"""
        return time.time() - self._state_change_time
    
    def is_ai_turn(self) -> bool:
        """Check if it's currently the AI's turn to speak"""
        return self.conversation_state in (
            ConversationState.GREETING,
            ConversationState.AI_RESPONDING
        )
    
    def is_user_turn(self) -> bool:
        """Check if it's currently the user's turn to speak"""
        return self.conversation_state in (
            ConversationState.LISTENING,
            ConversationState.USER_SPEAKING
        )
    
    def on_ai_start_speaking(self):
        """Called when AI starts generating/speaking a response"""
        if self.conversation_state == ConversationState.GREETING:
            pass  # Stay in greeting state
        else:
            self.set_conversation_state(ConversationState.AI_RESPONDING, "AI started speaking")
        self.assistant_speaking = True
        self.audio_output_active = True
        # PROFESSIONAL: Track response latency
        self.quality_tracker.on_ai_start_response()
    
    def on_ai_stop_speaking(self):
        """Called when AI finishes speaking"""
        self.assistant_speaking = False
        self.assistant_stop_time = time.time()
        self.audio_output_active = False
        
        # Transition to listening for user response
        if self.conversation_state not in (ConversationState.CALL_ENDING, ConversationState.ENDED):
            self.set_conversation_state(ConversationState.LISTENING, "AI finished speaking")
            self.eot_detector.reset_for_new_turn()
    
    def on_user_start_speaking(self):
        """Called when user starts speaking"""
        if self.conversation_state == ConversationState.LISTENING:
            self.set_conversation_state(ConversationState.USER_SPEAKING, "User started speaking")
            # ⏱️ LATENCY TRACKING: Start tracking new utterance
            self.latency_tracker.start_utterance()
        elif self.is_ai_turn():
            # User interrupted AI
            self.set_conversation_state(ConversationState.INTERRUPTED, "User interrupted AI")
    
    def on_user_stop_speaking(self):
        """Called when user stops speaking (end of turn detected)"""
        if self.conversation_state == ConversationState.USER_SPEAKING:
            self.set_conversation_state(ConversationState.PROCESSING, "User stopped speaking")
            # PROFESSIONAL: Track for latency measurement
            self.quality_tracker.on_user_end_of_turn()
            # ⏱️ LATENCY TRACKING: Mark user end of turn
            self.latency_tracker.mark_user_end_of_turn()
    
    def on_call_answered(self):
        """Called when the call is answered (SIP 200 OK received)"""
        self.call_answered = True
        self.set_conversation_state(ConversationState.GREETING, "Call answered")
        # Start AMD detection
        self.amd.start_detection()
        # Reset VAD statistics for new call
        self.vad.reset_statistics()
        # Reset audio preprocessor noise floor for fresh estimation
        self.audio_preprocessor.reset_noise_floor()
    
    def on_call_ended(self, outcome: str = "completed"):
        """Called when the call ends"""
        self.set_conversation_state(ConversationState.ENDED, f"Call ended: {outcome}")
        
        # PROFESSIONAL: Update quality tracker and log summary
        if self.amd_result is not None:
            self.quality_tracker.set_amd_result(self.amd_result)
        self.quality_tracker.set_call_outcome(outcome)
        self.quality_tracker.log_summary()
        
        # ⏱️ LATENCY TRACKING: Log final latency summary with bottleneck analysis
        self.latency_tracker.log_final_summary()
        
    def process_incoming_audio(self, audio_data: bytes, payload_type: int, timestamp: int):
        """
        Process incoming RTP audio packet in full-duplex mode.
        
        PROFESSIONAL COLD CALLING ENHANCEMENTS:
        - AMD detection in first 5 seconds
        - State-aware processing
        - Adaptive silence detection
        - Better turn-taking through state machine
        """
        try:
            # --- 1. DECODE AUDIO ---
            # Increment packet counter for RTP activity tracking
            self.packets_received += 1
            
            # ⏱️ LATENCY TRACKING: Mark RTP packet received from SIM gate
            self.latency_tracker.mark_rtp_receive(len(audio_data))
            
            # Convert payload based on type
            if payload_type == 0:  # PCMU/μ-law
                pcm_data = self.ulaw_to_pcm(audio_data)
            elif payload_type == 8:  # PCMA/A-law  
                pcm_data = self.alaw_to_pcm(audio_data)
            elif payload_type == 13:  # CN (Comfort Noise)
                # Generate silence for CN packets to keep stream alive
                # Standard frame is 20ms (160 samples at 8kHz)
                pcm_data = b'\x00' * 320 
            else:
                # Assume it's already PCM or unknown
                pcm_data = audio_data
            
            # ⏱️ LATENCY TRACKING: Mark decode complete
            self.latency_tracker.mark_decode_complete()
            
            # Validate PCM data length to prevent filter corruption
            if len(pcm_data) < 10:
                return
            
            # Record the ORIGINAL audio (before preprocessing) for authentic recordings
            if self.call_recorder:
                self.call_recorder.record_incoming_audio(pcm_data)
            
            # --- 2. PROFESSIONAL AUDIO PREPROCESSING ---
            # Apply 4-stage pipeline: Bandpass → Noise Suppression → AGC → Limiter
            # This cleans the audio before VAD and Gemini for better recognition
            pcm_data = self.audio_preprocessor.process_audio(pcm_data)
            
            # ⏱️ LATENCY TRACKING: Mark preprocessing complete
            self.latency_tracker.mark_preprocess_complete()
            
            # --- 3. ENHANCED VAD WITH METRICS ---
            # Use process_with_metrics for professional cold calling features
            # VAD now operates on cleaned audio for more accurate speech detection
            vad_result = self.vad.process_with_metrics(pcm_data)
            is_speech_active = vad_result['is_speech']
            filtered_audio = vad_result['audio']
            frame_energy = vad_result['energy']
            
            # ⏱️ LATENCY TRACKING: Mark VAD complete
            self.latency_tracker.mark_vad_complete(is_speech_active)
            
            # --- 4. PROFESSIONAL: AMD Detection (first 5 seconds) ---
            if self.amd.is_detecting():
                amd_result = self.amd.feed_audio(is_speech_active, frame_energy)
                if amd_result is not None and not self._amd_action_taken:
                    self._amd_action_taken = True
                    self.amd_result = amd_result
                    if amd_result:
                        logger.warning(f"📞 AMD: ANSWERING MACHINE DETECTED - consider leaving voicemail or hanging up")
                        # Optional: You could trigger voicemail mode or hangup here
                        # self._trigger_voicemail_mode() or self._trigger_hangup()
                    else:
                        logger.info(f"📞 AMD: HUMAN DETECTED - continuing with live conversation")
            
            # --- 5. PROFESSIONAL: Adaptive Silence Threshold ---
            adaptive_threshold = self.adaptive_silence.update(frame_energy, is_speech_active)
            # Update the legacy threshold for compatibility
            self._speech_energy_threshold = adaptive_threshold
            
            # --- 6. PROFESSIONAL: State Machine Updates ---
            if is_speech_active and len(filtered_audio) > 0:
                # User is speaking
                if self.conversation_state == ConversationState.LISTENING:
                    self.on_user_start_speaking()
                elif self.is_ai_turn():
                    # User interrupted AI - handled by Gemini server-side
                    pass
            
            # --- 7. Filter silence (don't send to Gemini) ---
            if not is_speech_active and len(filtered_audio) == 0:
                # Silence or background noise - feed to EOT detector but don't send to Gemini
                self.eot_detector.update(False, self.conversation_state)
                return
            
            # User is speaking - update voice activity time
            self.last_voice_activity_time = time.time()
            
            # Use the filtered audio (frames aligned to 20ms) instead of raw packet
            processed_pcm = filtered_audio
            
            # --- 8. PROFESSIONAL: Debounced EOT Detection ---
            # NOTE: Disabled separate eot_detector - using inline detection below instead
            # This prevents duplicate EOT signals and conflicting state machine updates
            # The eot_detector is still updated for tracking but doesn't trigger state changes
            self.eot_detector.update(is_speech_active, self.conversation_state)
            # DON'T call on_user_stop_speaking() here - it's called when EOT is actually sent
            
            # ⚡ LATENCY FIX: Compute EOT decision HERE (not in async send loop)
            # This eliminates duplicate processing that was causing 450ms+ delays
            should_send_eot = False
            current_time = time.time()
            
            # Initialize last_eot_time if not set
            if not hasattr(self, '_last_eot_time'):
                self._last_eot_time = 0
            
            # Check if we're in a state where EOT detection makes sense
            if self.conversation_state in (ConversationState.USER_SPEAKING, 
                                           ConversationState.LISTENING,
                                           ConversationState.PROCESSING):
                
                is_silence = self.adaptive_silence.is_silence(frame_energy)
                
                if is_silence:
                    # Track when silence started
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time
                        
                    silence_duration = current_time - self._silence_start_time
                    
                    # ⚡ EOT COOLDOWN: Don't send another EOT within 5s of the last one
                    # Increased from 2.5s to prevent multiple EOTs causing Gemini to restart
                    time_since_last_eot = current_time - self._last_eot_time
                    eot_cooldown_ok = time_since_last_eot >= 5.0
                    
                    # Only send EOT if:
                    # 1. Silence duration exceeds threshold (350ms)
                    # 2. We haven't already sent EOT (strict, no cooldown bypass)
                    # 3. We've detected speech since the last EOT (prevent spamming)
                    # 4. EOT cooldown has passed (5s between EOTs - increased from 2.5s)
                    if (silence_duration >= self._silence_threshold_sec and 
                        not self._end_of_turn_sent and
                        getattr(self, '_speech_detected_this_turn', False) and
                        eot_cooldown_ok):
                        should_send_eot = True
                        self._end_of_turn_sent = True
                        self._is_speaking_turn = False
                        self._speech_detected_this_turn = False
                        self._last_eot_time = current_time  # Update cooldown timer
                        # ⏱️ LATENCY FIX: DON'T call on_user_stop_speaking() here!
                        # It will be called when EOT is actually sent to Gemini
                        logger.debug(f"🔇 EOT computed: silence={silence_duration:.2f}s, state={self.conversation_state.name}")
                else:
                    # User is speaking - reset silence tracking but NOT eot_sent flag
                    # ⚡ FIX: Only reset _end_of_turn_sent if enough time has passed
                    # This prevents rapid EOT→speech→EOT cycles
                    if self._silence_start_time is not None:
                        self._silence_start_time = None
                        # Only allow new EOT if 5+ seconds since last EOT (increased from 2.5s)
                        if current_time - self._last_eot_time >= 5.0:
                            self._end_of_turn_sent = False
                        self._last_speech_time = current_time
                    
                    # Mark that we've detected speech this turn
                    self._speech_detected_this_turn = True
                    
                    # Update turn start time if we were previously silent
                    if not hasattr(self, '_is_speaking_turn') or not self._is_speaking_turn:
                        self._is_speaking_turn = True
                        self._turn_start_time = current_time
                
                # --- MAX TURN DURATION (Safety Net) ---
                if hasattr(self, '_is_speaking_turn') and self._is_speaking_turn:
                    turn_duration = current_time - self._turn_start_time
                    if turn_duration > 8.0:  # 8 seconds max
                        logger.warning(f"⚠️ Max turn duration reached (8s) - forcing EOT")
                        should_send_eot = True
                        self._end_of_turn_sent = True
                        self._last_eot_time = current_time
                        self._turn_start_time = current_time
            else:
                # AI is speaking or call is ending - reset EOT state
                self._silence_start_time = None
                self._end_of_turn_sent = False
                self._speech_detected_this_turn = False
            
            # Log voice activity periodically (for debugging)
            if not hasattr(self, '_last_voice_log_time') or (time.time() - self._last_voice_log_time) > 5.0:
                self._last_voice_log_time = time.time()
                logger.debug(f"🗣️ Voice activity in {self.session_id} | State: {self.conversation_state.name} | Energy: {frame_energy:.0f}")
            
            # Log once to show audio is flowing
            if not hasattr(self, '_audio_packet_count'):
                self._audio_packet_count = 0
                logger.info(f"🎤 Full-duplex streaming active - WebRTC VAD filtered audio sent to Gemini")
            
            # Start asyncio thread if not already running
            if not self.input_processing and not self.asyncio_thread_started:
                self.input_processing = True
                self.asyncio_thread_started = True  # Set flag before starting thread
                self.asyncio_thread = threading.Thread(target=self._run_asyncio_thread, daemon=True)
                self.asyncio_thread.start()
                # Give it a moment to initialize
                time.sleep(0.05)  # Reduced from 0.1s for faster startup
            
            # Add to buffer and send when we have enough for efficient transmission
            with self.buffer_lock:
                self.audio_buffer += processed_pcm
                
                # ⚡ OPTIMIZED: Send in smaller chunks (20-40ms) for lower latency
                # 20ms at 8kHz = 160 samples * 2 bytes = 320 bytes
                min_chunk = 320 
                
                if len(self.audio_buffer) >= min_chunk:
                    chunk_to_send = self.audio_buffer
                    self.audio_buffer = b""  # Clear buffer
                    
                    # ⚡ LATENCY FIX: Queue audio WITH pre-computed EOT flag
                    # This eliminates all duplicate processing in the async send loop
                    self.audio_input_queue.put((chunk_to_send, should_send_eot))
                    
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
    
    def _run_asyncio_thread(self):
        """Run dedicated asyncio thread for all Gemini communication"""
        # Create and set event loop for this thread
        self.asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_loop)
        
        try:
            # Run the main async loop
            self.asyncio_loop.run_until_complete(self._async_main_loop())
        except Exception as e:
            logger.error(f"Error in asyncio thread: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            # Close loop safely (check for None)
            if self.asyncio_loop and not self.asyncio_loop.is_closed():
                try:
                    self.asyncio_loop.close()
                except Exception as e:
                    logger.warning(f"Error closing asyncio loop: {e}")
            self.input_processing = False
    
    async def _async_main_loop(self):
        """Main async loop - handles initialization, sending, and receiving"""
        try:
            # Initialize voice session in this event loop (so websocket works properly)
            if not await self.voice_session.initialize_voice_session():
                logger.error("Failed to initialize voice session")
                return
            
            logger.info("🎤 Voice session initialized - starting audio streaming")
            
            # Wait a bit for WebSocket to fully stabilize before starting audio transmission
            await asyncio.sleep(0.2)
            
            # Start continuous receiver task
            receiver_task = asyncio.create_task(self._continuous_receive_responses())
            
            # Give receiver time to start
            await asyncio.sleep(0.1)
            
            # ✅ NUDGE MECHANISM: If gemini_greeting is enabled, send a nudge to make Gemini speak first
            # BUT ONLY AFTER THE CALL IS ANSWERED (SIP 200 OK received)
            custom_config = getattr(self.voice_session, 'custom_config', {}) or {}
            if custom_config.get('gemini_greeting', False):
                logger.info("🎙️ Gemini greeting enabled - waiting for call to be answered before sending nudge...")
                
                # Wait for call to be answered (max 60 seconds)
                wait_start = time.time()
                max_wait = 60.0  # Max time to wait for answer
                while not self.call_answered and self.input_processing:
                    await asyncio.sleep(0.1)  # Check every 100ms
                    if time.time() - wait_start > max_wait:
                        logger.warning("⚠️ Timeout waiting for call to be answered - not sending greeting nudge")
                        break
                
                # Only send nudge if call was actually answered
                if self.call_answered:
                    logger.info("📞 Call answered! Sending nudge to start conversation...")
                    try:
                        # --- FIX 3: Enable greeting protection ---
                        # This blinds the bot to user audio (Hello?) for 4 seconds while it starts speaking
                        self.greeting_protection_active = True
                        logger.info("🛡️ Greeting protection ENABLED - blocking user audio for 4s")
                        
                        # Background task to disable protection after 4 seconds
                        def disable_protection():
                            time.sleep(4)  # Wait for greeting to likely start playing
                            self.greeting_protection_active = False
                            logger.info("🛡️ Greeting protection DISABLED - listening to user now")
                        
                        threading.Thread(target=disable_protection, daemon=True).start()
                        # --- END FIX 3 ---
                        
                        # Send a brief text message and signal end_of_turn to trigger Gemini to speak
                        # This tells Gemini "it's your turn, start talking"
                        await self.voice_session.gemini_session.send(
                            input="[START CONVERSATION]",
                            end_of_turn=True  # Critical: signals Gemini to respond
                        )
                        logger.info("✅ Nudge sent - Gemini should start speaking now")
                    except Exception as nudge_error:
                        logger.warning(f"⚠️ Nudge failed, Gemini may not speak first: {nudge_error}")
                        # Make sure to disable protection if nudge fails
                        self.greeting_protection_active = False
                else:
                    logger.warning("⚠️ Call not answered - skipping greeting nudge")
            
            # ⚡ LATENCY-OPTIMIZED Main loop: process audio from queue and send to Gemini
            # Key optimizations:
            # 1. Audio is queued as (bytes, should_send_eot) - no duplicate processing
            # 2. Drain queue aggressively - process ALL available chunks immediately
            # 3. Minimal async sleep - just enough to yield control
            audio_chunks_sent = 0
            last_send_time = time.time()
            
            # Batch processing for efficiency - process up to 20 chunks at once
            # At 50 packets/sec, this processes up to 400ms of audio per iteration
            max_batch_size = 20  # ⚡ Increased from 5 to 20 for better throughput
            
            while self.input_processing:
                try:
                    # ⚡ AGGRESSIVE queue drain - collect ALL available audio immediately
                    chunks_collected = 0
                    pending_sends = []  # List of (processed_audio, should_send_eot)
                    
                    while chunks_collected < max_batch_size:
                        try:
                            # ⚡ Queue now contains (audio_bytes, should_send_eot) tuples
                            queue_item = self.audio_input_queue.get_nowait()
                            
                            # Handle both old format (just bytes) and new format (tuple)
                            if isinstance(queue_item, tuple):
                                audio_chunk, should_send_eot = queue_item
                            else:
                                audio_chunk = queue_item
                                should_send_eot = False
                            
                            pending_sends.append((audio_chunk, should_send_eot))
                            chunks_collected += 1
                        except queue.Empty:
                            break
                    
                    # ⚡ Send ALL collected chunks to Gemini - fast path with pre-computed EOT
                    if pending_sends and self.voice_session and self.voice_session.gemini_session:
                        for audio_chunk, should_send_eot in pending_sends:
                            await self._send_audio_to_gemini_fast(audio_chunk, should_send_eot)
                            audio_chunks_sent += 1
                        last_send_time = time.time()
                    
                    # ⚡ REMOVED: No more silence sending - it adds unnecessary latency
                    # Gemini handles silence detection server-side
                    
                    # ⚡ Minimal yield - just enough for receiver task to run
                    await asyncio.sleep(0.0005)  # 0.5ms yield (was 1ms) - even faster
                        
                except Exception as e:
                    logger.error(f"Error in main audio loop: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    await asyncio.sleep(0.05)  # Brief pause before retry
            
            # Clean up receiver task
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            logger.error(f"Error in async main loop: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _calculate_audio_energy(self, audio_data: bytes) -> float:
        """Calculate RMS energy of audio data for voice activity detection"""
        try:
            # Convert bytes to int16 samples
            import struct
            num_samples = len(audio_data) // 2
            if num_samples == 0:
                return 0.0
            samples = struct.unpack(f'{num_samples}h', audio_data[:num_samples * 2])
            # Calculate RMS
            sum_squares = sum(s * s for s in samples)
            rms = (sum_squares / num_samples) ** 0.5
            return rms
        except Exception:
            return 0.0
    
    async def _send_audio_to_gemini_fast(self, audio_chunk: bytes, should_send_eot: bool = False):
        """
        ⚡ LATENCY-OPTIMIZED: Ultra-fast audio send to Gemini.
        
        This method does MINIMAL processing - all VAD, energy calculation,
        state machine updates, and EOT detection are already done in 
        process_incoming_audio() before the audio hits the queue.
        
        Key optimizations:
        1. No duplicate VAD/energy calculation (saves ~1-3ms per packet)
        2. Pre-computed EOT flag (saves ~5ms of state machine logic)
        3. Minimal timeout (0.5s instead of 1s)
        4. Fast path for common case (no EOT)
        """
        try:
            if not self.voice_session or not self.voice_session.gemini_session:
                return
            
            # --- GREETING PROTECTION ---
            if getattr(self, 'greeting_protection_active', False):
                return
            
            # ⚡ Convert telephony audio to Gemini format (this is required)
            processed_audio = self.voice_session.convert_telephony_to_gemini(audio_chunk)
            
            # Initialize counters if needed
            if not hasattr(self, '_audio_send_count'):
                self._audio_send_count = 0
                self._audio_send_bytes = 0
                self._last_send_log_time = time.time()
                self._send_latency_samples = []
                self._slow_send_count = 0
            
            self._audio_send_count += 1
            self._audio_send_bytes += len(processed_audio)
            
            # ⚡ FAST SEND: Use pre-computed EOT flag from queue
            try:
                send_start = time.time()
                await asyncio.wait_for(
                    self.voice_session.gemini_session.send(
                        input={"data": processed_audio, "mime_type": "audio/pcm;rate=24000"},
                        end_of_turn=should_send_eot
                    ),
                    timeout=0.5  # ⚡ Reduced from 1.0s to 0.5s for faster failure
                )
                send_duration = time.time() - send_start
                
                # ⏱️ LATENCY TRACKING: Mark audio sent to Gemini
                self.latency_tracker.mark_gemini_send(len(processed_audio))
                
                # Track latency samples (keep last 100)
                if not hasattr(self, '_send_latency_samples'):
                    self._send_latency_samples = []
                self._send_latency_samples.append(send_duration)
                if len(self._send_latency_samples) > 100:
                    self._send_latency_samples.pop(0)
                
                # Log if send took longer than expected
                if send_duration > 0.03:  # ⚡ Reduced threshold from 50ms to 30ms
                    if not hasattr(self, '_slow_send_count'):
                        self._slow_send_count = 0
                    self._slow_send_count += 1
                    if self._slow_send_count <= 5 or self._slow_send_count % 50 == 0:
                        logger.warning(f"⏱️ Slow Gemini send #{self._slow_send_count}: {send_duration*1000:.0f}ms")
                
                # Log EOT if sent and update state machine + latency tracking
                if should_send_eot:
                    logger.info(f"✅ End-of-turn sent (state: {self.conversation_state.name})")
                    # ⏱️ LATENCY FIX: Mark EOT when ACTUALLY sent to Gemini (not when detected)
                    # on_user_stop_speaking() also calls latency_tracker.mark_user_end_of_turn()
                    self.on_user_stop_speaking()
                    
            except asyncio.TimeoutError:
                if not hasattr(self, '_send_timeout_count'):
                    self._send_timeout_count = 0
                self._send_timeout_count += 1
                if self._send_timeout_count % 10 == 1:
                    logger.warning(f"⚠️ Gemini send timeout (count: {self._send_timeout_count})")
                return
            except Exception as send_err:
                error_str = str(send_err)
                if "no close frame" in error_str.lower():
                    if not hasattr(self, '_connection_closed_logged'):
                        logger.warning("⚠️ Gemini WebSocket connection closed unexpectedly")
                        self._connection_closed_logged = True
                return
            
            # Log every 50 packets with latency stats
            current_time = time.time()
            if self._audio_send_count % 50 == 0:
                elapsed = current_time - self._last_send_log_time
                avg_latency = sum(self._send_latency_samples) / len(self._send_latency_samples) if self._send_latency_samples else 0
                max_latency = max(self._send_latency_samples) if self._send_latency_samples else 0
                
                # Calculate time since last response for NO_RESP warning
                time_since_response = current_time - getattr(self, '_last_response_time', current_time)
                
                status_parts = [
                    f"📤 Sent {self._audio_send_count} pkts",
                    f"elapsed={elapsed:.2f}s",
                    f"avg_lat={avg_latency*1000:.1f}ms",
                    f"max_lat={max_latency*1000:.1f}ms",
                    f"slow={self._slow_send_count}",
                ]
                
                if time_since_response > 5:
                    status_parts.append(f"⚠️ NO_RESP={time_since_response:.0f}s")
                
                logger.info(" | ".join(status_parts))
                self._last_send_log_time = time.time()
                
        except Exception as e:
            logger.error(f"Error in _send_audio_to_gemini_fast: {e}")
    
    async def _send_audio_to_gemini(self, audio_chunk: bytes):
        """
        Send audio chunk to Gemini with professional cold calling turn-taking.
        
        PROFESSIONAL ENHANCEMENTS:
        - State-aware end-of-turn detection (doesn't trigger during AI speaking)
        - Adaptive silence threshold (adjusts to call noise level)
        - Debounced triggering (requires confirmation frames)
        - Greeting protection (blocks user audio during initial greeting)
        - Max turn duration safety net (prevents buffer freeze)
        """
        try:
            if not self.voice_session or not self.voice_session.gemini_session:
                return
            
            # --- GREETING PROTECTION ---
            # Block user audio during initial greeting to prevent interruption
            if getattr(self, 'greeting_protection_active', False):
                return
            
            # Convert telephony audio to Gemini format
            processed_audio = self.voice_session.convert_telephony_to_gemini(audio_chunk)
            
            # Initialize latency tracking if needed
            if not hasattr(self, '_audio_send_count'):
                self._audio_send_count = 0
                self._audio_send_bytes = 0
                self._last_send_log_time = time.time()
                self._send_latency_samples = []
                self._slow_send_count = 0
                self._turn_start_time = time.time()
            
            self._audio_send_count += 1
            self._audio_send_bytes += len(processed_audio)
            
            current_time = time.time()
            
            # ============================================================
            # PROFESSIONAL COLD CALLING: STATE-AWARE END-OF-TURN DETECTION
            # Uses adaptive thresholds and debouncing for reliable turn-taking
            # ============================================================
            
            audio_energy = self._calculate_audio_energy(audio_chunk)
            
            # Use adaptive threshold from the AdaptiveSilenceDetector
            is_silence = self.adaptive_silence.is_silence(audio_energy)
            
            # PROFESSIONAL: Only detect end-of-turn when appropriate
            # Don't send end_of_turn signals when:
            # 1. AI is speaking (GREETING, AI_RESPONDING)
            # 2. Call is ending
            # 3. We haven't detected speech since last end-of-turn
            should_send_eot = False
            
            # Check if we're in a state where EOT detection makes sense
            if self.conversation_state in (ConversationState.USER_SPEAKING, 
                                           ConversationState.LISTENING,
                                           ConversationState.PROCESSING):
                
                if is_silence:
                    # Track when silence started
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time
                        
                    # PROFESSIONAL: Use longer threshold (400ms) and require confirmation
                    silence_duration = current_time - self._silence_start_time
                    
                    # Only send EOT if:
                    # 1. Silence duration exceeds threshold (400ms)
                    # 2. We haven't already sent EOT
                    # 3. We've detected speech since the last EOT (prevent spamming)
                    if (silence_duration >= self._silence_threshold_sec and 
                        not self._end_of_turn_sent and
                        getattr(self, '_speech_detected_this_turn', False)):
                        should_send_eot = True
                        logger.debug(f"🔇 Silence detected ({silence_duration:.2f}s) - state: {self.conversation_state.name}")
                else:
                    # User is speaking - reset silence tracking
                    if self._silence_start_time is not None:
                        self._silence_start_time = None
                        self._end_of_turn_sent = False
                        self._last_speech_time = current_time
                    
                    # Mark that we've detected speech this turn
                    self._speech_detected_this_turn = True
                    
                    # Update turn start time if we were previously silent
                    if not hasattr(self, '_is_speaking_turn') or not self._is_speaking_turn:
                        self._is_speaking_turn = True
                        self._turn_start_time = current_time
                
                # --- MAX TURN DURATION (Safety Net) ---
                # If speech continues for > 8 seconds without a break, force a flush
                # Increased from 5s to 8s for more natural long utterances
                if hasattr(self, '_is_speaking_turn') and self._is_speaking_turn:
                    turn_duration = current_time - self._turn_start_time
                    if turn_duration > 8.0:  # 8 seconds max (increased from 5s)
                        logger.warning(f"⚠️ Max turn duration reached (8s) - forcing end_of_turn")
                        should_send_eot = True
                        self._turn_start_time = current_time
            else:
                # AI is speaking or call is ending - reset EOT state
                self._silence_start_time = None
                self._end_of_turn_sent = False
                self._speech_detected_this_turn = False
            
            # Send end-of-turn signal if triggered
            if should_send_eot and not self._end_of_turn_sent:
                self._end_of_turn_sent = True
                self._is_speaking_turn = False
                self._speech_detected_this_turn = False
                
                # Update conversation state
                self.on_user_stop_speaking()
                
                # Send with end_of_turn=True
                try:
                    await asyncio.wait_for(
                        self.voice_session.gemini_session.send(
                            input={"data": processed_audio, "mime_type": "audio/pcm;rate=24000"},
                            end_of_turn=True
                        ),
                        timeout=1.0
                    )
                    logger.info(f"✅ End-of-turn sent (state: {self.conversation_state.name})")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send end_of_turn: {e}")
            
            # Send normal audio
            try:
                send_start = time.time()
                await asyncio.wait_for(
                    self.voice_session.gemini_session.send(
                        input={"data": processed_audio, "mime_type": "audio/pcm;rate=24000"},
                        end_of_turn=False
                    ),
                    timeout=1.0  # Reduced timeout to 1s for faster failure
                )
                send_duration = time.time() - send_start
                
                # ⏱️ LATENCY TRACKING: Mark audio sent to Gemini
                self.latency_tracker.mark_gemini_send(len(processed_audio))
                
                # Track latency samples (keep last 100)
                if not hasattr(self, '_send_latency_samples'):
                    self._send_latency_samples = []
                self._send_latency_samples.append(send_duration)
                if len(self._send_latency_samples) > 100:
                    self._send_latency_samples.pop(0)
                
                # Log if send took longer than expected
                if send_duration > 0.05:  # 50ms threshold
                    self._slow_send_count += 1
                    if self._slow_send_count <= 5 or self._slow_send_count % 20 == 0:
                        logger.warning(f"⏱️ Slow Gemini send #{self._slow_send_count}: {send_duration*1000:.0f}ms")
            
            except asyncio.TimeoutError:
                # Fail silently on timeouts to keep stream moving
                if not hasattr(self, '_send_timeout_count'):
                    self._send_timeout_count = 0
                self._send_timeout_count += 1
                if self._send_timeout_count % 10 == 1:
                    logger.warning(f"⚠️ Gemini send timeout (count: {self._send_timeout_count})")
                return
            except Exception as send_err:
                error_str = str(send_err)
                if "no close frame" in error_str.lower():
                    if not hasattr(self, '_connection_closed_logged'):
                        logger.warning("⚠️ Gemini WebSocket connection closed unexpectedly")
                        self._connection_closed_logged = True
                return
            
            # Log every 50 packets with latency stats
            if self._audio_send_count % 50 == 0:
                elapsed = time.time() - self._last_send_log_time
                avg_latency = sum(self._send_latency_samples) / len(self._send_latency_samples) if self._send_latency_samples else 0
                max_latency = max(self._send_latency_samples) if self._send_latency_samples else 0
                
                # Calculate time since last response for NO_RESP warning
                time_since_response = current_time - getattr(self, '_last_response_time', current_time)
                
                # Build status string
                status_parts = [
                    f"📤 Sent {self._audio_send_count} pkts",
                    f"elapsed={elapsed:.2f}s",
                    f"avg_lat={avg_latency*1000:.1f}ms",
                    f"max_lat={max_latency*1000:.1f}ms",
                    f"slow={self._slow_send_count}",
                ]
                
                # Add NO_RESP warning if no response for > 5 seconds
                if time_since_response > 5:
                    status_parts.append(f"⚠️ NO_RESP={time_since_response:.0f}s")
                
                logger.info(" | ".join(status_parts))
                self._last_send_log_time = time.time()
                
        except Exception as e:
            logger.error(f"Error in _send_audio_to_gemini: {e}")

    def buffer_and_play(self, pcm_data: bytes):
        """
        Ultra Low Latency Jitter Buffer - starts playback immediately.
        
        Gemini 24kHz -> Resampled 8kHz -> Minimal Buffer -> Playback Queue
        """
        # If already playing, send immediately - no additional buffering
        if self.playback_started:
            self._send_audio_immediate(pcm_data)
            return

        # Initial buffering phase - collect minimal audio before starting
        self.jitter_buffer += pcm_data
        
        # LATENCY FIX: Start playback immediately if we have ANY meaningful audio (>20ms)
        # or if the received chunk was large (Gemini burst)
        if len(self.jitter_buffer) >= self.jitter_buffer_threshold:
            self.playback_started = True
            
            # Send all buffered audio immediately - no logging to save I/O time
            self._send_audio_immediate(self.jitter_buffer)
            self.jitter_buffer = b""
            
    async def _continuous_receive_responses(self):
        """
        Continuously receive responses from Gemini in a separate task.
        
        PROFESSIONAL COLD CALLING ENHANCEMENTS:
        - State machine integration (on_ai_start_speaking, on_ai_stop_speaking)
        - AMD detection via transcripts
        - Better turn completion handling
        """
        logger.info("🎧 Starting continuous response receiver")
        
        while self.input_processing and self.voice_session.gemini_session:
            try:
                # Double-check session is still active before receiving
                if not self.input_processing or not self.voice_session.gemini_session:
                    break
                
                # Get response from Gemini - this is a continuous stream
                turn = self.voice_session.gemini_session.receive()
                
                turn_had_content = False
                turn_had_audio = False  # Track if we received audio this turn
                
                # Debug flag to log response attributes once
                if not hasattr(self, '_response_attrs_logged'):
                    self._response_attrs_logged = False
                    
                async for response in turn:
                    if not self.input_processing:
                        break
                    
                    # Log response attributes once for debugging
                    if not self._response_attrs_logged:
                        attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                        logger.info(f"🔍 Response object attributes: {', '.join(attrs[:20])}")  # First 20 to avoid clutter
                        
                        # Deep inspection of the response object
                        logger.info(f"🔬 Response type: {type(response)}")
                        logger.info(f"🔬 Response.__dict__ keys: {list(response.__dict__.keys()) if hasattr(response, '__dict__') else 'No __dict__'}")
                        
                        # Try to dump the whole response
                        try:
                            if hasattr(response, 'model_dump'):
                                dump = response.model_dump()
                                logger.info(f"🔬 Response.model_dump() keys: {list(dump.keys())}")
                                # Show first few items of each key
                                for key, value in dump.items():
                                    if value is not None:
                                        logger.info(f"🔬   {key}: {str(value)[:200]}")
                        except Exception as e:
                            logger.debug(f"Could not dump response: {e}")
                        
                        self._response_attrs_logged = True
                        
                    try:
                        # Check for audio data in the response (new API format)
                        if hasattr(response, 'data') and response.data:
                            turn_had_content = True
                            audio_bytes = response.data
                            if isinstance(audio_bytes, bytes):
                                # PROFESSIONAL: Update state machine on first audio
                                if not turn_had_audio:
                                    turn_had_audio = True
                                    self.on_ai_start_speaking()
                                    # ⏱️ LATENCY TRACKING: Mark first audio from Gemini
                                    self.latency_tracker.mark_gemini_first_response(len(audio_bytes))
                                
                                # Log audio reception for debugging
                                if not hasattr(self, '_audio_recv_count'):
                                    self._audio_recv_count = 0
                                    self._audio_recv_bytes = 0
                                
                                self._audio_recv_count += 1
                                self._audio_recv_bytes += len(audio_bytes)
                                
                                # Track last response time to detect Gemini freezes
                                self._last_response_time = time.time()
                                
                                # Reset end-of-turn state when Gemini responds
                                # This prepares for the next user turn detection
                                if hasattr(self, '_end_of_turn_sent') and self._end_of_turn_sent:
                                    self._end_of_turn_sent = False
                                    self._silence_start_time = None
                                    self._speech_detected_this_turn = False
                                    logger.debug("🔄 End-of-turn reset - Gemini is responding")
                                
                                # Log every 10 audio chunks received
                                if self._audio_recv_count % 10 == 1:
                                    logger.info(f"📥 Received {self._audio_recv_count} audio chunks ({self._audio_recv_bytes} bytes total) from Gemini")
                                
                                # Convert from 24kHz to 8kHz telephony format
                                telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_bytes)
                                # ⏱️ LATENCY TRACKING: Mark conversion complete
                                self.latency_tracker.mark_convert_complete()
                                # Buffer and play for smoothness
                                self.buffer_and_play(telephony_audio)
                            elif isinstance(audio_bytes, str):
                                try:
                                    # Decode base64 if needed
                                    decoded = base64.b64decode(audio_bytes)
                                    telephony_audio = self.voice_session.convert_gemini_to_telephony(decoded)
                                    self.buffer_and_play(telephony_audio)
                                except:
                                    pass
                        
                        # Handle text response (for logging only, should be minimal with audio-only mode)
                        if hasattr(response, 'text') and response.text:
                            turn_had_content = True
                            # Only log if it's not thinking/reasoning text
                            text = response.text.strip()
                            if text and not text.startswith('**'):  # Skip markdown thinking text
                                self.voice_session.session_logger.log_transcript(
                                    "assistant_response", text
                                )
                                logger.info(f"AI Response: {text}")
                        
                        # Feed transcripts to goodbye detector AND AMD detector
                        try:
                            # Check if we have server_content with transcriptions
                            if hasattr(response, "server_content") and response.server_content:
                                sc = response.server_content
                                
                                # 1) User (input) transcript
                                if hasattr(sc, "input_transcription") and sc.input_transcription:
                                    if hasattr(sc.input_transcription, "text") and sc.input_transcription.text:
                                        user_txt = sc.input_transcription.text.strip()
                                        if user_txt and user_txt != '.':  # Filter out noise/silence markers
                                            logger.info(f"🎤 User transcript: {user_txt}")
                                            self.goodbye.feed(user_txt, "user", self._trigger_hangup, self._cancel_hangup)
                                            
                                            # PROFESSIONAL: Feed to AMD detector
                                            if self.amd.is_detecting():
                                                self.amd.feed_transcript(user_txt)
                                
                                # 2) Agent (output) transcript
                                if hasattr(sc, "output_transcription") and sc.output_transcription:
                                    if hasattr(sc.output_transcription, "text") and sc.output_transcription.text:
                                        agent_txt = sc.output_transcription.text.strip()
                                        # Filter out thinking text (starts with **) and noise markers
                                        if agent_txt and not agent_txt.startswith('**') and agent_txt != '.':
                                            logger.info(f"🗣️ Agent transcript: {agent_txt}")
                                            self.goodbye.feed(agent_txt, "agent", self._trigger_hangup, self._cancel_hangup)
                                        elif agent_txt and agent_txt.startswith('**'):
                                            logger.debug(f"💭 Agent thinking: {agent_txt[:100]}...")
                                
                                # 3) ⚡ INSTANT INTERRUPTION HANDLING - stop model speech immediately
                                if hasattr(sc, "interrupted") and sc.interrupted:
                                    logger.info(f"🔄 User interrupted model in session {self.session_id} - stopping speech immediately!")
                                    
                                    # PROFESSIONAL: Update state machine and track quality
                                    self.set_conversation_state(ConversationState.INTERRUPTED, "User interrupted")
                                    self.quality_tracker.on_user_interrupt()
                                    
                                    # ACTIVATE KILL SWITCH
                                    self.is_interrupted = True
                                    
                                    # Clear Jitter Buffer and Reset Playback
                                    self.playback_started = False
                                    self.jitter_buffer = b""
                                    
                                    self._cancel_hangup()
                                    self.audio_output_active = False
                                    self.last_output_time = 0
                                    
                                    # Aggressive flush
                                    try:
                                        while not self.output_queue.empty():
                                            self.output_queue.get_nowait()
                                    except:
                                        pass
                                    
                        except Exception as e:
                            logger.debug(f"Error checking transcripts: {e}")
                            
                    except Exception as e:
                        if self.input_processing:
                            logger.error(f"Error processing response item: {e}")
                        # Continue processing other responses
                        continue
                
                # After turn completes, prepare for next turn
                if turn_had_content:
                    logger.info(f"🔄 Turn complete (had content) - state: {self.conversation_state.name}")
                    
                    # PROFESSIONAL: Update state machine - AI finished speaking
                    if turn_had_audio:
                        self.on_ai_stop_speaking()
                    
                    # Reset silence tracking so we can detect next user silence
                    self._silence_start_time = None
                    self._end_of_turn_sent = False
                    self._speech_detected_this_turn = False
                    
                    # Reset EOT detector for new turn
                    self.eot_detector.reset_for_new_turn()
                else:
                    logger.debug("Empty turn received, continuing to listen")
                
                # Reset interrupted flag after turn completes to allow next response
                if self.is_interrupted:
                    logger.debug("🔄 Resetting interrupted flag after turn complete")
                    self.is_interrupted = False
                    # PROFESSIONAL: Transition from INTERRUPTED to LISTENING
                    if self.conversation_state == ConversationState.INTERRUPTED:
                        self.set_conversation_state(ConversationState.LISTENING, "Interruption handled")
                
                # Small yield to prevent tight loop
                await asyncio.sleep(0.001)
                        
            except asyncio.CancelledError:
                # Clean shutdown
                logger.info("🎧 Receiver task cancelled")
                break
            except Exception as e:
                if self.input_processing:  # Only log if we're still processing
                    # Check if it's a connection close error (expected during shutdown)
                    error_str = str(e).lower()
                    if 'close frame' in error_str or 'connection closed' in error_str or 'concurrency' in error_str:
                        logger.debug(f"Connection closing or concurrent access: {e}")
                        break  # Exit gracefully
                    else:
                        logger.error(f"Error receiving from Gemini: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        await asyncio.sleep(0.1)  # Brief pause before retry
                else:
                    break  # Exit if not processing anymore
                    
        logger.info("🎧 Stopped continuous response receiver")
    
    def _send_audio_immediate(self, audio_data: bytes):
        """Send audio data immediately in optimized chunks for real-time streaming"""
        if not audio_data:
            return
            
        # Send in 20ms chunks for optimal real-time performance (standard for VoIP)
        chunk_size = 320  # 20ms at 8kHz = 160 samples * 2 bytes = 320 bytes
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            # Send each chunk immediately to the output queue
            self.send_audio(chunk)
    
    # Removed _process_chunk - we use continuous streaming instead
    
    def send_audio(self, pcm16: bytes):
        """
        Takes 16-bit mono PCM at 8000 Hz, encodes to u-law using C-lib, queues.
        """
        if self.call_recorder:
            self.call_recorder.record_outgoing_audio(pcm16)
        
        # Direct conversion using C library - extremely fast
        # Removed soft_limit checks for pure speed
        ulaw = audioop.lin2ulaw(pcm16, 2) 
        
        packet_size = 160 # 20ms
        for i in range(0, len(ulaw), packet_size):
            self.output_queue.put(ulaw[i:i + packet_size])
            
        self.latency_tracker.mark_rtp_send(len(pcm16))
    
    def play_greeting_file(self, greeting_file: str = "greeting.wav"):
        """Play a greeting WAV file at the start of the call and return estimated duration"""
        try:
            # Check if it's an absolute path or relative
            if os.path.isabs(greeting_file):
                greeting_path = greeting_file
            else:
                greeting_path = greeting_file
                
            if not os.path.exists(greeting_path):
                logger.warning(f"Greeting file {greeting_path} not found, trying default greeting.wav")
                # Fall back to default greeting
                greeting_path = "greeting.wav"
                if not os.path.exists(greeting_path):
                    logger.warning(f"Default greeting file not found either, skipping greeting")
                    return 0.0
            
            # Check if RTP session is ready
            if not self.output_processing:
                logger.warning("⚠️ RTP output processing not ready, cannot play greeting")
                return 0.0
                
            logger.info(f"🎵 Playing greeting file: {greeting_path}")
            logger.info(f"📞 RTP session state: output_processing={self.output_processing}, remote_addr={self.remote_addr}")
            
            # Check file extension
            file_ext = os.path.splitext(greeting_path)[1].lower()
            
            if file_ext == '.mp3':
                # Handle MP3 file
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(greeting_path)
                    # Convert to WAV format in memory
                    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    wav_data = audio.raw_data
                    frames = len(wav_data) // 2  # 16-bit samples
                    sample_rate = 8000
                    channels = 1
                    sample_width = 2
                    audio_data = wav_data
                    duration_seconds = frames / sample_rate
                    logger.info(f"Converted MP3 to PCM: {frames} frames, {duration_seconds:.2f}s")
                except ImportError:
                    logger.error("❌ pydub not installed - cannot play MP3 files")
                    logger.error("💡 Install pydub: pip install pydub")
                    return 0.0
            else:
                # Load WAV file
                with wave.open(greeting_path, 'rb') as wav_file:
                    # Get WAV parameters
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    
                    # Calculate duration in seconds
                    duration_seconds = frames / sample_rate if sample_rate > 0 else 0.0
                    
                    logger.info(f"Greeting file info: {frames} frames, {sample_rate}Hz, {channels}ch, {sample_width}bytes/sample")
                    logger.info(f"Estimated duration: {duration_seconds:.2f} seconds")
                    
                    # Read all audio data
                    audio_data = wav_file.readframes(frames)
                
                # Convert to mono if stereo
                if channels == 2:
                    audio_data = audioop.tomono(audio_data, sample_width, 1, 1)
                    logger.info("Converted stereo to mono")
                
                # Convert to 16-bit if needed
                if sample_width == 1:
                    audio_data = audioop.bias(audio_data, 1, -128)  # Convert unsigned to signed
                    audio_data = audioop.lin2lin(audio_data, 1, 2)  # Convert to 16-bit
                elif sample_width == 3:
                    audio_data = audioop.lin2lin(audio_data, 3, 2)  # Convert 24-bit to 16-bit
                elif sample_width == 4:
                    audio_data = audioop.lin2lin(audio_data, 4, 2)  # Convert 32-bit to 16-bit
                
                # Resample to 8kHz if needed
                if sample_rate != 8000:
                    import numpy as np
                    # Resample using audioop
                    # ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0)
                    audio_data, _ = audioop.ratecv(audio_data, 2, 1, sample_rate, 8000, None)
                    
                    target_samples = len(audio_data) // 2
                    duration_seconds = target_samples / 8000.0
                    
                    logger.info(f"Resampled from {sample_rate}Hz to 8000Hz, new duration: {duration_seconds:.2f}s")
                
                # Send the greeting audio
                logger.info(f"🎙️ Sending greeting audio: {len(audio_data)} bytes")
                self.send_audio(audio_data)
                
                logger.info("✅ Greeting played successfully")
                return duration_seconds
                
        except Exception as e:
            logger.error(f"Error playing greeting file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue without greeting - don't let this break the call
            return 0.0
    
    def _process_output_queue(self):
        """
        Dequeue G.711 payloads and transmit with a high-precision, self-correcting
        RTP pacing loop for smooth, real-time voice with minimal latency.
        """
        import time  # Ensure time is available
        import queue # Ensure queue is available (for queue.Empty)
        
        ptime_seconds = 0.020  # 20ms packet interval in seconds
        
        # Use perf_counter for high-resolution timing
        try:
            # Wait for the first packet to arrive to set the start time
            payload = self.output_queue.get(timeout=0.5)  # Reduced from 1.0s
            # Mark that we're actively outputting audio
            self.audio_output_active = True
            self.last_output_time = time.time()
            # Reset inactivity timer when agent starts speaking
            self.last_voice_activity_time = time.time()
        except queue.Empty:
            # No audio for 0.5 second, just start the loop
            pass
        except Exception:
            pass # Handle other errors
            
        start_time = time.perf_counter()
        next_packet_time = start_time
        
        while self.output_processing:
            try:
                # Reset interruption flag if queue was empty (start of new phrase)
                if self.output_queue.empty():
                    self.is_interrupted = False

                payload = self.output_queue.get(timeout=0.040)
                
                # KILL SWITCH CHECK: If interrupted, drop this packet and skip sleep
                if self.is_interrupted:
                    continue

                # We got audio - mark as active and reset inactivity timer
                self.audio_output_active = True
                self.last_output_time = time.time()
                # Reset inactivity timer when agent speaks (so we only count user inactivity)
                self.last_voice_activity_time = time.time()
            except queue.Empty:
                # No audio in queue - assistant stopped speaking
                if self.audio_output_active:
                    self.audio_output_active = False
                    self.assistant_stop_time = time.time()
                    logger.debug(f"🔊 Assistant stopped speaking - enabling grace period")
                    # DISABLED QUEUE CLEARING: This was deleting user speech if they spoke/interrupted 
                    # right as the assistant stopped. We rely on Gemini's server-side echo cancellation now.
                    
                    # with self.buffer_lock:
                    #     if self.audio_buffer:
                    #         logger.debug(f"🧹 Clearing {len(self.audio_buffer)} bytes of buffered audio")
                    #         self.audio_buffer = b""
                    # # Also clear the input queue
                    # try:
                    #     cleared_count = 0
                    #     while not self.audio_input_queue.empty():
                    #         self.audio_input_queue.get_nowait()
                    #         cleared_count += 1
                    #     if cleared_count > 0:
                    #         logger.debug(f"🧹 Cleared {cleared_count} queued audio chunks")
                    # except:
                    #     pass
                # Reset the pacer to avoid building up "missed" time
                next_packet_time = time.perf_counter() + ptime_seconds
                continue
            except Exception:
                continue # Other queue errors

            # Transmit the packet
            self._send_rtp(payload)
            
            # --- FIX START: PREVENT CATCH-UP LOOP ---
            next_packet_time += ptime_seconds
            current_time = time.perf_counter()
            sleep_duration = next_packet_time - current_time
            
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                # We are lagging - just reset the clock immediately.
                # Don't try to catch up as it causes CPU hogging and audio glitches.
                next_packet_time = current_time + ptime_seconds
                # Add a tiny sleep to yield CPU to the input thread
                time.sleep(0.001) 
            # --- FIX END ---
    
    def _send_rtp(self, payload: bytes):
        """Send RTP packet with payload"""
        rtp_packet = self._create_rtp_packet(payload, payload_type=0)
        self.rtp_socket.sendto(rtp_packet, self.remote_addr)
    
    def _sleep_ms(self, ms: int):
        """Sleep for specified milliseconds"""
        import time
        time.sleep(ms / 1000.0)
    
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
    
    def soft_limit_16bit(self, pcm: bytes, ratio: float = 0.95) -> bytes:
        """Ultra-light limiter to avoid μ-law harshness on peaks - scales samples simply and fast"""
        return audioop.mul(pcm, 2, ratio)
    
    def pcm_to_ulaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to μ-law"""
        try:
            # Apply ultra-light soft limiting to prevent μ-law harshness on peaks
            # pcm_data = self.soft_limit_16bit(pcm_data, 0.95) # REMOVED: was causing clipping distortion
            
            # Convert to μ-law
            return audioop.lin2ulaw(pcm_data, 2)
        except Exception as e:
            logger.error(f"Error in PCM to μ-law conversion: {e}")
            # Fallback to simple conversion
            return audioop.lin2ulaw(pcm_data, 2)
    
    def _monitor_voice_timeout(self):
        """Monitor voice activity and end call if no USER voice detected for timeout period
        
        Only counts inactivity when it's the user's turn to speak (not during agent responses).
        Timer is reset when:
        - User speaks (real voice detected, not background noise)
        - Agent starts speaking (so we don't count agent response time)
        """
        logger.info(f"⏱️ Voice timeout monitor started for session {self.session_id}")
        
        while self.timeout_monitoring and self.output_processing:
            try:
                # Only count inactivity when it's user's turn (agent not speaking)
                if not self.audio_output_active:
                    # Check how long since last voice activity
                    time_since_voice = time.time() - self.last_voice_activity_time
                    
                    # Log periodically for debugging (only when it's user's turn)
                    if int(time_since_voice) % 10 == 0 and int(time_since_voice) > 0:
                        remaining = self.voice_timeout_seconds - time_since_voice
                        if remaining > 0:
                            logger.debug(f"⏳ No user voice for {int(time_since_voice)}s (user's turn), timeout in {int(remaining)}s")
                    
                    # Check if timeout exceeded
                    if time_since_voice > self.voice_timeout_seconds:
                        logger.warning(f"⏱️ No user voice activity for {int(time_since_voice)}s in session {self.session_id}")
                        logger.warning(f"🔚 Ending call due to user inactivity timeout")
                        
                        # Stop monitoring to prevent multiple BYE messages
                        self.timeout_monitoring = False
                        
                        # Send BYE to end the call
                        self._send_bye_to_gate()
                        break
                else:
                    # Agent is speaking - don't count this time as inactivity
                    # The timer is already being reset in _process_output_queue
                    pass
                
                # Check every 1 second
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in voice timeout monitor: {e}")
                time.sleep(1.0)
        
        logger.info(f"⏱️ Voice timeout monitor stopped for session {self.session_id}")
    
    def _vad_barge_in_thread(self):
        """
        Dedicated VAD thread for barge-in detection.
        Runs independently of audio processing to interrupt playback immediately when user speaks.
        This enables natural conversation flow where users can interrupt the AI mid-sentence.
        """
        logger.info(f"🎤 VAD barge-in thread started for session {self.session_id}")
        
        consecutive_speech_frames = 0
        min_frames_for_interrupt = 3  # ~60ms at 20ms frames - quick but filters noise
        
        while self.vad_thread_running and self.output_processing:
            try:
                # Get audio from VAD queue with timeout
                try:
                    pcm_data = self.vad_queue.get(timeout=0.05)  # 50ms timeout
                except queue.Empty:
                    # No audio, reset speech counter
                    if consecutive_speech_frames > 0:
                        consecutive_speech_frames = max(0, consecutive_speech_frames - 1)
                    continue
                
                # Skip VAD check if we're not playing audio (no need to interrupt)
                if not self.audio_output_active and self.output_queue.empty():
                    consecutive_speech_frames = 0
                    continue
                
                # Run VAD analysis
                is_speech = self.vad.is_speech(pcm_data)
                
                if is_speech:
                    consecutive_speech_frames += 1
                    
                    if not self._user_speaking:
                        self._user_speaking = True
                        self._user_speech_start_time = time.time()
                        logger.debug(f"🎤 User started speaking (VAD thread)")
                    
                    # Check if we should trigger barge-in
                    if consecutive_speech_frames >= min_frames_for_interrupt:
                        speech_duration_ms = (time.time() - self._user_speech_start_time) * 1000
                        
                        if speech_duration_ms >= self._barge_in_threshold_ms:
                            # Only trigger barge-in if AI is actively speaking
                            if self.audio_output_active or not self.output_queue.empty():
                                logger.info(f"🔄 BARGE-IN detected! User speaking for {speech_duration_ms:.0f}ms - interrupting playback")
                                
                                # Set interruption flag
                                self.is_interrupted = True
                                
                                # Clear jitter buffer
                                self.playback_started = False
                                self.jitter_buffer = b""
                                
                                # Aggressive queue flush
                                try:
                                    cleared = 0
                                    while not self.output_queue.empty():
                                        self.output_queue.get_nowait()
                                        cleared += 1
                                    if cleared > 0:
                                        logger.debug(f"🧹 Cleared {cleared} audio packets from output queue")
                                except:
                                    pass
                                
                                # Mark output as inactive
                                self.audio_output_active = False
                                self.last_output_time = 0
                                
                                # Reset speech tracking after barge-in
                                consecutive_speech_frames = 0
                                self._user_speaking = False
                else:
                    # Gradual decay of speech counter
                    consecutive_speech_frames = max(0, consecutive_speech_frames - 1)
                    if consecutive_speech_frames == 0:
                        self._user_speaking = False
                
            except Exception as e:
                logger.error(f"Error in VAD barge-in thread: {e}")
                time.sleep(0.01)
        
        logger.info(f"🎤 VAD barge-in thread stopped for session {self.session_id}")
    
    def _trigger_hangup(self):
        """Trigger graceful hangup when goodbye is detected"""
        if self._bye_sent:
            return
        self._bye_sent = True
        logger.info("👋 Goodbye confirmed → sending SIP BYE")
        self._send_bye_to_gate()

    def _cancel_hangup(self):
        """Cancel pending hangup due to new speech"""
        # called if barge-in or clarifying speech happens
        logger.debug("🛑 Hangup grace window canceled due to new speech")
    
    def _send_bye_to_gate(self):
        """Send SIP BYE message to Gate to terminate the call"""
        try:
            # Use cleanup lock to prevent concurrent cleanup
            with self._cleanup_lock:
                # Check if cleanup already done
                if self._cleanup_done:
                    logger.debug(f"Cleanup already performed for session {self.session_id}, skipping")
                    return
                
                # Mark cleanup as in progress
                self._cleanup_done = True
                
                logger.info(f"📞 Sending SIP BYE to terminate session {self.session_id}")
                
                # Check if we have stored SIP details in the voice session
                if not self.voice_session or not self.voice_session.sip_handler:
                    # Try to get from active_sessions as fallback
                    from __main__ import active_sessions
                    
                    session_data = active_sessions.get(self.session_id)
                    if not session_data:
                        logger.warning(f"⚠️ Session {self.session_id} not found - cannot send BYE (already terminated)")
                        # Session already removed, just stop local processing
                        self.cleanup_threads()
                        return
                    
                    # Get SIP details from session data
                    sip_handler = session_data.get('sip_handler')
                    call_id = session_data.get('call_id', self.session_id)
                    from_tag = session_data.get('from_tag', 'voice-agent')
                    to_tag = session_data.get('to_tag', '')
                    called_number = session_data.get('called_number', 'unknown')
                else:
                    # Use stored SIP details from voice session
                    sip_handler = self.voice_session.sip_handler
                    call_id = self.voice_session.call_id
                    from_tag = self.voice_session.from_tag
                    to_tag = self.voice_session.to_tag
                    called_number = self.voice_session.called_number
                
                if not sip_handler:
                    logger.warning(f"⚠️ No SIP handler found for session {self.session_id}")
                    # Clean up locally
                    self.cleanup_threads()
                    return
                
                # Create BYE message
                username = sip_handler.config.get('username', '200')
                bye_message = f"""BYE sip:{called_number}@{sip_handler.gate_ip}:{sip_handler.sip_port} SIP/2.0
Via: SIP/2.0/UDP {sip_handler.local_ip}:{sip_handler.sip_port};branch=z9hG4bK-bye-{uuid.uuid4().hex[:8]}
From: <sip:{username}@{sip_handler.gate_ip}>;tag={from_tag}
To: <sip:{called_number}@{sip_handler.gate_ip}>;tag={to_tag}
Call-ID: {call_id}
CSeq: 999 BYE
Max-Forwards: 70
User-Agent: WindowsVoiceAgent/1.0
Content-Length: 0

"""
                
                # Send BYE message
                gate_addr = (sip_handler.gate_ip, sip_handler.sip_port)
                sip_handler.socket.sendto(bye_message.encode(), gate_addr)
                logger.info(f"✅ BYE message sent to {gate_addr} for session {self.session_id}")
                
                # Stop recording FIRST to ensure mixed file is created before cleanup
                if self.voice_session and hasattr(self.voice_session, 'call_recorder') and self.voice_session.call_recorder:
                    try:
                        self.voice_session.call_recorder.stop_recording()
                        logger.info(f"📼 Recording stopped for session {self.session_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error stopping recording: {e}")
                
                # Remove from active sessions - DO THIS BEFORE calling cleanup_threads
                # to avoid race conditions with asyncio loop shutdown
                from __main__ import active_sessions
                logger.info(f"🔍 Checking active_sessions before removal: session in dict = {self.session_id in active_sessions}, total sessions = {len(active_sessions)}")
                
                if self.session_id in active_sessions:
                    del active_sessions[self.session_id]
                    logger.info(f"🗑️ Session {self.session_id} removed from active sessions")
                else:
                    logger.warning(f"⚠️ Session {self.session_id} was NOT in active_sessions during cleanup - may have been removed already by BYE handler")
                
                logger.info(f"🔍 After removal: total sessions = {len(active_sessions)}")
                
                # Remove RTP session (this will call cleanup_threads automatically)
                sip_handler.rtp_server.remove_session(self.session_id)
                
                logger.info(f"✅ Session {self.session_id} full cleanup complete after sending BYE")
                
        except Exception as e:
            logger.error(f"Error sending BYE message: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Clean up locally on error
            self.cleanup_threads()
    
    def cleanup_threads(self):
        """Properly cleanup all RTP session threads and resources"""
        logger.info(f"🧹 Cleaning up RTP session {self.session_id} threads...")
        
        # Stop all processing flags first
        self.input_processing = False
        self.output_processing = False
        self.processing = False
        self.timeout_monitoring = False
        self.vad_thread_running = False  # Stop VAD barge-in thread
        
        # Clear queues to unblock threads
        try:
            while not self.audio_input_queue.empty():
                self.audio_input_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        except:
            pass
        
        # Clear VAD queue
        try:
            while not self.vad_queue.empty():
                self.vad_queue.get_nowait()
        except:
            pass
        
        # Close asyncio loop if it exists
        if self.asyncio_loop and not self.asyncio_loop.is_closed():
            try:
                # Cancel all pending tasks before stopping the loop
                if self.asyncio_loop.is_running():
                    # Get all pending tasks and cancel them
                    def cancel_all_tasks():
                        tasks = [t for t in asyncio.all_tasks(self.asyncio_loop) if not t.done()]
                        for task in tasks:
                            task.cancel()
                        logger.info(f"🛑 Cancelled {len(tasks)} pending asyncio task(s) for session {self.session_id}")
                        return len(tasks)
                    
                    # Schedule task cancellation in the loop's thread
                    future = asyncio.run_coroutine_threadsafe(
                        asyncio.create_task(asyncio.sleep(0)),  # Dummy coroutine to run cancel_all_tasks
                        self.asyncio_loop
                    )
                    try:
                        future.result(timeout=0.2)
                    except:
                        pass
                    
                    # Now stop the loop
                    self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
                    time.sleep(0.15)  # Give it time to stop
                
                self.asyncio_loop.close()
                logger.info(f"✅ Closed asyncio loop for session {self.session_id}")
            except Exception as e:
                logger.warning(f"⚠️ Error closing asyncio loop: {e}")
        
        # Wait briefly for threads to finish (with timeout)
        threads_to_check = []
        if self.output_thread and self.output_thread.is_alive():
            threads_to_check.append(("output", self.output_thread))
        if self.asyncio_thread and self.asyncio_thread.is_alive():
            threads_to_check.append(("asyncio", self.asyncio_thread))
        if self.timeout_monitor_thread and self.timeout_monitor_thread.is_alive():
            threads_to_check.append(("timeout", self.timeout_monitor_thread))
        
        # Give threads 500ms total to finish
        start_time = time.time()
        while threads_to_check and (time.time() - start_time) < 0.5:
            threads_to_check = [(name, thread) for name, thread in threads_to_check if thread.is_alive()]
            if threads_to_check:
                time.sleep(0.05)
        
        if threads_to_check:
            logger.warning(f"⚠️ {len(threads_to_check)} thread(s) still running after cleanup: {[name for name, _ in threads_to_check]}")
        else:
            logger.info(f"✅ All threads stopped for session {self.session_id}")
        
        # Clear references
        self.voice_session = None
        self.call_recorder = None
        self.rtp_socket = None  # Don't close the socket, just remove reference
        
        logger.info(f"✅ RTP session {self.session_id} cleanup complete")
    

def get_default_owner_id() -> int:
    """Get the default owner ID for incoming calls (first admin user)"""
    db = get_session()
    try:
        # Try to get first admin user
        admin_user = db.query(User).filter(User.role == UserRole.ADMIN).first()
        if admin_user:
            return admin_user.id
        
        # If no admin, try to get superadmin
        superadmin_user = db.query(User).filter(User.role == UserRole.SUPERADMIN).first()
        if superadmin_user:
            return superadmin_user.id
        
        # If no users at all, return 1 as fallback (should not happen in production)
        logger.warning("⚠️  No admin or superadmin user found for incoming call, using default owner_id=1")
        return 1
    except Exception as e:
        logger.error(f"Error getting default owner: {e}")
        return 1  # Fallback
    finally:
        db.close()

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
        
        # Extension registration for outbound calls
        self.extension_registered = False
        self.registration_call_id = None
        
        # Store pending outbound calls for authentication
        self.pending_invites = {}  # call_id -> invite_info
        
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
            
            # Note: We act as SIP trunk for INCOMING calls (Gate VoIP registers with us)
            logger.info("🎯 Acting as SIP trunk - waiting for Gate VoIP to register with us")
            
            # Also register as Extension 200 for OUTBOUND calls
            logger.info("📞 Registering as Extension 200 for outbound calling capability...")
            self._register_as_extension()
            
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
                elif first_line.startswith('SIP/2.0 100'):
                    message_type = "100 Trying (Call Initiated)"
                elif first_line.startswith('SIP/2.0 180'):
                    message_type = "180 Ringing (Phone Ringing)"
                elif first_line.startswith('SIP/2.0 183'):
                    message_type = "183 Session Progress (Ringing with Media)"
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
                elif first_line.startswith('SIP/2.0'):
                    # Handle all SIP responses (100, 180, 183, 200, 401, 403, 404, etc.)
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
            
            # Extract Asterisk headers from incoming INVITE
            asterisk_headers = self._extract_asterisk_headers(message)
            
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
            
            logger.info(f"📞 Incoming call from {caller_id}")
            
            # Print Asterisk linkedid if available
            if asterisk_headers.get('linkedid'):
                logger.info(f"🔗 Asterisk Linked ID (Call-Wide): {asterisk_headers['linkedid']}")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Generate to_tag for this session (used in SIP responses)
            to_tag = session_id[:8]  # Use first 8 chars of session_id
            
            # Send 180 Ringing first
            ringing_response = self._create_sip_ringing_response(message, session_id)
            self.socket.sendto(ringing_response.encode(), addr)
            logger.info("🔔 Sent 180 Ringing")
            
            # Send 200 OK response
            ok_response = self._create_sip_ok_response(message, session_id)
            self.socket.sendto(ok_response.encode(), addr)
            logger.info("✅ Sent 200 OK")
            
            # Create or update CRM database - call answered
            db = get_session()
            try:
                crm_session = db.query(CallSession).filter(CallSession.session_id == session_id).first()
                if crm_session:
                    crm_session.status = CallStatus.ANSWERED
                    # Update asterisk_linkedid if available
                    if asterisk_headers.get('linkedid'):
                        crm_session.asterisk_linkedid = asterisk_headers.get('linkedid')
                else:
                    # Create new session for incoming calls
                    # Get default owner for incoming calls (first admin user)
                    default_owner_id = get_default_owner_id()
                    
                    # Try to find lead by caller phone number
                    lead = db.query(Lead).filter(
                        (Lead.phone == caller_id) |
                        (Lead.phone == caller_id.replace('+', '')) |
                        (Lead.phone.contains(caller_id[-10:] if len(caller_id) >= 10 else caller_id))  # Last 10 digits
                    ).first()
                    
                    crm_session = CallSession(
                        session_id=session_id,
                        caller_id=caller_id,
                        called_number=config.get('phone_number', 'Unknown'),
                        lead_id=lead.id if lead else None,
                        owner_id=default_owner_id,  # Assign to default admin user
                        status=CallStatus.ANSWERED,
                        started_at=datetime.utcnow(),
                        answered_at=datetime.utcnow(),
                        asterisk_linkedid=asterisk_headers.get('linkedid')  # Store Asterisk Linked ID
                    )
                    db.add(crm_session)
                db.commit()
                
                # Log the saved linkedid
                if asterisk_headers.get('linkedid'):
                    logger.info(f"💾 Saved Asterisk Linked ID to database: {asterisk_headers.get('linkedid')}")
            except Exception as e:
                logger.error(f"Error updating CRM session status: {e}")
            finally:
                db.close()
            
            # Detect caller's country and language
            caller_country = detect_caller_country(caller_id)
            language_info = get_language_config(caller_country)
            voice_config = create_voice_config(language_info)
            
            logger.info(f"🌍 Language: {language_info['lang']} ({caller_country})")
            
            # Get the user's API key for this call
            user_api_key = None
            try:
                from crm_database_mongodb import UserManager
                db = get_session()
                user_manager = UserManager(db)
                user = user_manager.get_user_by_id(default_owner_id)
                if user and user.google_api_key:
                    user_api_key = user.google_api_key
                    logger.info(f"🔑 Retrieved API key for user {user.username} (ID: {default_owner_id})")
                else:
                    logger.warning(f"⚠️ No API key found for user ID {default_owner_id}, using default")
                db.close()
            except Exception as e:
                logger.error(f"❌ Error retrieving user API key: {e}")
            
            # Create voice session with language-specific configuration and user's API key
            voice_session = WindowsVoiceSession(session_id, caller_id, self.phone_number, voice_config, custom_config=None, api_key=user_api_key)
            
            # Store SIP details in voice session for sending BYE later
            voice_session.sip_handler = self
            voice_session.from_tag = from_tag
            voice_session.to_tag = to_tag
            voice_session.call_id = call_id
            
            # Create RTP session for audio
            rtp_session = self.rtp_server.create_session(session_id, addr, voice_session, voice_session.call_recorder)
            
            active_sessions[session_id] = {
                "voice_session": voice_session,
                "rtp_session": rtp_session,
                "caller_addr": addr,
                "status": "waiting_for_ack",
                "call_start": datetime.now(timezone.utc),
                "call_id": call_id,
                "sip_handler": self,
                "from_tag": from_tag,
                "to_tag": to_tag,
                "caller_id": caller_id,
                "called_number": self.phone_number,
                "asterisk_linkedid": asterisk_headers.get('linkedid')  # Store Asterisk Linked ID
            }
            
            logger.info(f"⏳ Waiting for ACK to establish call {session_id}...")
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
        """Handle call termination with immediate cleanup for next call readiness"""
        try:
            # Find session for this address
            session_found = False
            for session_id, session_data in list(active_sessions.items()):
                if session_data.get("caller_addr") == addr:
                    session_found = True
                    logger.info(f"📞 Call termination received for session {session_id}")
                    
                    rtp_session = session_data.get("rtp_session")
                    
                    # Use cleanup lock to prevent concurrent cleanup with timeout monitor
                    if rtp_session and hasattr(rtp_session, '_cleanup_lock'):
                        with rtp_session._cleanup_lock:
                            # Check if cleanup already done
                            if hasattr(rtp_session, '_cleanup_done') and rtp_session._cleanup_done:
                                logger.debug(f"Cleanup already performed for session {session_id}, skipping")
                                # Still send OK response
                                ok_response = "SIP/2.0 200 OK\r\n\r\n"
                                self.socket.sendto(ok_response.encode(), addr)
                                return
                            
                            # Mark cleanup as done
                            if hasattr(rtp_session, '_cleanup_done'):
                                rtp_session._cleanup_done = True
                            
                            # Send 200 OK response immediately
                            ok_response = "SIP/2.0 200 OK\r\n\r\n"
                            self.socket.sendto(ok_response.encode(), addr)
                            logger.info(f"✅ Sent BYE 200 OK response")
                            
                            # Immediate session cleanup for next call readiness
                            voice_session = session_data["voice_session"]
                            
                            # Stop all processing immediately using cleanup_threads
                            if rtp_session:
                                # Note: cleanup_threads will stop all processing and clean up resources
                                logger.info(f"🛑 Initiating RTP cleanup for session {session_id}")
                            
                            # Stop recording BEFORE removing from active_sessions to ensure mixed file is created
                            if voice_session and hasattr(voice_session, 'call_recorder') and voice_session.call_recorder:
                                try:
                                    voice_session.call_recorder.stop_recording()
                                    logger.info(f"📼 Recording stopped for session {session_id}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Error stopping recording: {e}")
                            
                            # Remove from active sessions immediately
                            del active_sessions[session_id]
                            logger.info(f"🗑️ Session {session_id} removed from active sessions")
                            
                            # Cleanup RTP session (this calls cleanup_threads)
                            self.rtp_server.remove_session(session_id)
                            
                            # Voice session cleanup (non-blocking)
                            try:
                                if hasattr(voice_session, 'cleanup'):
                                    voice_session.cleanup()
                            except Exception as e:
                                logger.warning(f"Voice session cleanup warning (non-critical): {e}")
                            
                            # Update CRM database asynchronously to avoid blocking
                            def update_crm_async():
                                try:
                                    from crm_database import CallStatus as CRMCallStatus
                                    db = get_session()
                                    try:
                                        crm_session = db.query(CallSession).filter(CallSession.session_id == session_id).first()
                                        if crm_session:
                                            crm_session.status = CRMCallStatus.COMPLETED
                                            crm_session.ended_at = datetime.utcnow()
                                            if crm_session.started_at and crm_session.ended_at:
                                                crm_session.duration = int((crm_session.ended_at - crm_session.started_at).total_seconds())
                                            crm_session.save()  # MongoDB requires explicit save
                                            logger.info(f"📋 CRM database updated for session {session_id}: status={crm_session.status}, duration={crm_session.duration}s")
                                    except Exception as e:
                                        logger.error(f"Error updating CRM session: {e}")
                                    finally:
                                        db.close()
                                except Exception as e:
                                    logger.error(f"Error in CRM update thread: {e}")
                            
                            # Run CRM update in background thread to avoid blocking
                            threading.Thread(target=update_crm_async, daemon=True).start()
                            
                            # System ready for next call
                            logger.info(f"✅ Call cleanup completed - system ready for next call")
                            logger.info(f"📞 Active sessions: {len(active_sessions)}")
                    else:
                        # No cleanup lock (old session), do regular cleanup
                        # Send 200 OK response immediately
                        ok_response = "SIP/2.0 200 OK\r\n\r\n"
                        self.socket.sendto(ok_response.encode(), addr)
                        logger.info(f"✅ Sent BYE 200 OK response")
                        
                        # Immediate session cleanup for next call readiness
                        voice_session = session_data["voice_session"]
                        
                        # Stop all processing immediately
                        if rtp_session:
                            logger.info(f"🛑 Initiating RTP cleanup for session {session_id} (legacy path)")
                        
                        # Stop recording BEFORE removing from active_sessions to ensure mixed file is created
                        if voice_session and hasattr(voice_session, 'call_recorder') and voice_session.call_recorder:
                            try:
                                voice_session.call_recorder.stop_recording()
                                logger.info(f"📼 Recording stopped for session {session_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error stopping recording: {e}")
                        
                        # Remove from active sessions immediately
                        del active_sessions[session_id]
                        logger.info(f"🗑️ Session {session_id} removed from active sessions")
                        
                        # Cleanup RTP session (this calls cleanup_threads if available)
                        self.rtp_server.remove_session(session_id)
                        
                        # Voice session cleanup (non-blocking)
                        try:
                            if hasattr(voice_session, 'cleanup'):
                                voice_session.cleanup()
                        except Exception as e:
                            logger.warning(f"Voice session cleanup warning (non-critical): {e}")
                    
                    break
            
            if not session_found:
                # Still send OK response even if session not found
                ok_response = "SIP/2.0 200 OK\r\n\r\n"
                self.socket.sendto(ok_response.encode(), addr)
                logger.warning(f"⚠️ BYE received from unknown address {addr}, sent OK anyway")
                    
        except Exception as e:
            logger.error(f"Error handling BYE: {e}")
            # Always try to send OK response
            try:
                ok_response = "SIP/2.0 200 OK\r\n\r\n"
                self.socket.sendto(ok_response.encode(), addr)
            except:
                pass
    
    def _handle_ack(self, message: str, addr):
        """Handle ACK message - call is now fully established"""
        try:
            # Extract Call-ID to find the session
            call_id = None
            for line in message.split('\n'):
                line = line.strip()
                if line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
                    break
            
            if not call_id:
                logger.debug("ACK received but no Call-ID found")
                return
            
            # Find the session with matching call_id
            session_id = None
            for sid, session_data in active_sessions.items():
                if session_data.get("call_id") == call_id and session_data.get("status") == "waiting_for_ack":
                    session_id = sid
                    break
            
            if not session_id:
                logger.debug(f"ACK received for Call-ID {call_id} but no matching session found")
                return
            
            logger.info(f"✅ ACK received - Call {session_id} is now fully established!")
            
            voice_session = active_sessions[session_id]["voice_session"]
            rtp_session = active_sessions[session_id]["rtp_session"]
            
            # Now start the RTP async loop which will connect to Gemini, then play greeting
            def start_voice_session():
                try:
                    # Start the RTP session's async processing loop (which will initialize Gemini)
                    if rtp_session and not rtp_session.asyncio_thread_started:
                        rtp_session.input_processing = True
                        rtp_session.asyncio_thread_started = True  # Set flag before starting thread
                        rtp_session.asyncio_thread = threading.Thread(target=rtp_session._run_asyncio_thread, daemon=True)
                        rtp_session.asyncio_thread.start()
                        logger.info(f"🎧 Started RTP async loop for session {session_id}")
                    elif rtp_session and rtp_session.asyncio_thread_started:
                        logger.info(f"🎧 RTP async loop already started for session {session_id}")
                    
                    # Wait for Gemini to connect (check every 0.5s, timeout after 30s)
                    logger.info(f"⏳ Waiting for Gemini to connect...")
                    max_wait_time = 30  # seconds
                    wait_interval = 0.5  # seconds
                    elapsed_time = 0
                    
                    while elapsed_time < max_wait_time:
                        if session_id not in active_sessions:
                            logger.warning("⚠️ Call ended while waiting for Gemini")
                            return
                        
                        # Check if Gemini session is ready
                        if voice_session.gemini_session:
                            logger.info(f"✅ Gemini connected after {elapsed_time:.1f}s")
                            break
                        
                        time.sleep(wait_interval)
                        elapsed_time += wait_interval
                    
                    if not voice_session.gemini_session:
                        logger.error(f"❌ Gemini failed to connect within {max_wait_time}s")
                        if session_id in active_sessions:
                            del active_sessions[session_id]
                        return
                    
                    # Mark as active *after* Gemini is ready
                    if session_id in active_sessions:
                        active_sessions[session_id]["status"] = "active"
                        logger.info(f"🎯 Voice session {session_id} is now active and ready")
                    
                    # Play greeting *after* Gemini is ready (unless gemini_greeting is enabled)
                    def play_greeting():
                        # Small delay to ensure audio pipeline is ready
                        time.sleep(0.5)
                        
                        # Check if call is still active
                        if session_id not in active_sessions:
                            logger.warning("⚠️ Call ended before greeting could be played")
                            return
                        
                        # ✅ Skip file-based greeting if gemini_greeting is enabled
                        custom_config = getattr(voice_session, 'custom_config', {}) or {}
                        if custom_config.get('gemini_greeting', False):
                            logger.info("🎙️ Gemini greeting enabled - skipping file-based greeting (Gemini will speak first)")
                            return
                        
                        logger.info("🎵 Playing greeting...")
                        
                        # Play greeting file
                        greeting_duration = rtp_session.play_greeting_file("greeting.wav")
                        
                        if greeting_duration > 0:
                            logger.info(f"✅ Greeting played ({greeting_duration:.1f}s). Ready for conversation.")
                        else:
                            logger.warning("⚠️ Greeting file not found or failed to play")
                    
                    threading.Thread(target=play_greeting, daemon=True).start()
                        
                except Exception as e:
                    logger.error(f"❌ Failed to start voice session: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Remove failed session
                    if session_id in active_sessions:
                        del active_sessions[session_id]
            
            threading.Thread(target=start_voice_session, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error handling ACK: {e}")
    
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
            logger.info("✅ Gate VoIP trunk registered")
            
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
        
        # Create optimized SDP with only codecs we actually send (removed G.729)
        sdp_content = f"""v=0
o=voice-agent {int(time.time())} {int(time.time())} IN IP4 {self.local_ip}
s=Voice Agent Session
c=IN IP4 {self.local_ip}
t=0 0
m=audio 5004 RTP/AVP 0 8 101
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:101 telephone-event/8000
a=fmtp:101 0-15
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
    
    def make_outbound_call(self, phone_number: str, custom_config: dict = None, gate_slot: int = None, owner_id: int = None) -> Optional[str]:
        """Initiate outbound call as registered Extension 200
        
        Args:
            phone_number: Phone number to call
            custom_config: Optional custom configuration for the call
            gate_slot: Gate slot number (9-19) to use for this call. If None, defaults to 10
            owner_id: User ID who owns this call (for API key assignment)
        
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            # Check if extension is registered
            if not self.extension_registered:
                logger.error("❌ Cannot make outbound call - Extension 200 not registered")
                logger.error("💡 Wait for extension registration to complete")
                return None
            
            # Determine which gate slot to use
            if gate_slot is None:
                gate_slot = 10  # Default to slot 10
                logger.info(f"⚠️ No gate slot specified, using default slot {gate_slot}")
            elif gate_slot < 9 or gate_slot > 19:
                logger.error(f"❌ Invalid gate slot {gate_slot}. Must be between 9 and 19")
                return None
            
            # Add gate slot prefix for SIM gate routing
            # Remove '+' and add gate slot prefix
            dialed_number = phone_number.replace('+', '').replace(' ', '').replace('-', '')
            if not dialed_number.startswith(str(gate_slot)):
                dialed_number = str(gate_slot) + dialed_number
            
            logger.info(f"📞 Initiating outbound call to {phone_number} (dialing: {dialed_number}) as Extension 200")
            logger.info(f"📞 Call will be routed through Gate VoIP → {self.config.get('outbound_trunk', 'gsm2')} trunk → Port {gate_slot}")
            
            session_id = str(uuid.uuid4())
            username = self.config.get('username', '200')
            
            # Create SDP for outbound call
            sdp_content = f"""v=0
o={username} {int(time.time())} {int(time.time())} IN IP4 {self.local_ip}
s=Voice Agent Outbound Call
c=IN IP4 {self.local_ip}
t=0 0
m=audio 5004 RTP/AVP 0 8 101
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:101 telephone-event/8000
a=fmtp:101 0-15
a=sendrecv
a=ptime:20"""
            
            # Create INVITE as Extension 200 using dialed_number with prefix
            invite_message = f"""INVITE sip:{dialed_number}@{self.gate_ip} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{session_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={session_id[:8]}
To: <sip:{dialed_number}@{self.gate_ip}>
Call-ID: {session_id}
CSeq: 1 INVITE
Contact: <sip:{username}@{self.local_ip}:{self.sip_port}>
User-Agent: WindowsVoiceAgent/1.0
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO
Supported: replaces, timer
Content-Type: application/sdp
Content-Length: {len(sdp_content)}

{sdp_content}"""
            
            # Store INVITE information for potential authentication challenge
            self.pending_invites[session_id] = {
                'phone_number': dialed_number,  # Store dialed number with prefix
                'original_number': phone_number,  # Store original for reference
                'session_id': session_id,
                'username': username,
                'sdp_content': sdp_content,
                'cseq': 1,
                'custom_config': custom_config,  # Store custom config for use in success handler
                'from_tag': session_id[:8],  # Store our own from_tag for sending BYE later
                'owner_id': owner_id  # Store owner_id for API key retrieval
            }
            
            # Send INVITE
            self.socket.sendto(invite_message.encode(), (self.gate_ip, self.sip_port))
            logger.info(f"📤 Sent INVITE for outbound call to {phone_number} (dialed: {dialed_number})")
            logger.debug(f"INVITE message:\n{invite_message}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Error making outbound call: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _register_as_extension(self):
        """Register as Extension 200 for outbound calling capability"""
        try:
            logger.info("🔗 Registering as Extension 200 for outbound calls...")
            logger.info(f"   Username: {self.config.get('username', '200')}")
            logger.info(f"   Gate IP: {self.gate_ip}")
            logger.info(f"   Local IP: {self.local_ip}")
            
            # Create REGISTER request for extension registration
            call_id = str(uuid.uuid4())
            self.registration_call_id = call_id
            username = self.config.get('username', '200')
            
            register_message = f"""REGISTER sip:{self.gate_ip} SIP/2.0
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
            logger.info(f"📡 Sent extension registration request to {self.gate_ip}:{self.sip_port}")
            logger.debug(f"REGISTER message:\n{register_message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register as extension: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
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
    
    def _create_authenticated_extension_register(self, challenge_response: str):
        """Create authenticated REGISTER for extension registration"""
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
            
            username = self.config.get('username', '200')
            password = self.config.get('password', '')
            
            logger.info(f"🔐 Creating authenticated extension REGISTER")
            logger.info(f"   Username: {username}")
            logger.info(f"   Realm: {realm}")
            
            # Create digest response
            import hashlib
            uri = f"sip:{self.gate_ip}"
            method = "REGISTER"
            
            # Calculate HA1
            ha1_input = f"{username}:{realm}:{password}"
            ha1 = hashlib.md5(ha1_input.encode()).hexdigest()
            
            # Calculate HA2  
            ha2_input = f"{method}:{uri}"
            ha2 = hashlib.md5(ha2_input.encode()).hexdigest()
            
            # Calculate response
            if qop:
                nc = "00000001"
                cnonce = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
                response_input = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
                response_hash = hashlib.md5(response_input.encode()).hexdigest()
            else:
                response_input = f"{ha1}:{nonce}:{ha2}"
                response_hash = hashlib.md5(response_input.encode()).hexdigest()
            
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
            
            # Create authenticated REGISTER for extension
            cseq_number = 2
            register_message = f"""REGISTER sip:{self.gate_ip} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{self.registration_call_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={self.registration_call_id[:8]}
To: <sip:{username}@{self.gate_ip}>
Call-ID: {self.registration_call_id}
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
            logger.info(f"🔑 Sent authenticated extension registration")
            
        except Exception as e:
            logger.error(f"❌ Failed to create authenticated extension REGISTER: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _create_authenticated_invite(self, challenge_response: str):
        """Create authenticated INVITE for outbound call"""
        try:
            # Extract Call-ID to find the pending invite
            call_id = None
            for line in challenge_response.split('\n'):
                line = line.strip()
                if line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
                    break
            
            if not call_id or call_id not in self.pending_invites:
                logger.error("❌ Cannot find pending INVITE for authentication")
                return
            
            invite_info = self.pending_invites[call_id]
            
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
            
            username = invite_info['username']
            password = self.config.get('password', '')
            phone_number = invite_info['phone_number']
            session_id = invite_info['session_id']
            sdp_content = invite_info['sdp_content']
            
            logger.info(f"🔐 Creating authenticated INVITE for outbound call to {phone_number}")
            
            # Create digest response
            import hashlib
            uri = f"sip:{phone_number}@{self.gate_ip}"
            method = "INVITE"
            
            # Calculate HA1
            ha1_input = f"{username}:{realm}:{password}"
            ha1 = hashlib.md5(ha1_input.encode()).hexdigest()
            
            # Calculate HA2
            ha2_input = f"{method}:{uri}"
            ha2 = hashlib.md5(ha2_input.encode()).hexdigest()
            
            # Calculate response
            if qop:
                nc = "00000001"
                cnonce = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
                response_input = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
                response_hash = hashlib.md5(response_input.encode()).hexdigest()
            else:
                response_input = f"{ha1}:{nonce}:{ha2}"
                response_hash = hashlib.md5(response_input.encode()).hexdigest()
            
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
            
            # Increment CSeq for retried INVITE
            new_cseq = invite_info['cseq'] + 1
            self.pending_invites[call_id]['cseq'] = new_cseq
            
            # Create authenticated INVITE
            authenticated_invite = f"""INVITE sip:{phone_number}@{self.gate_ip} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{session_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={session_id[:8]}
To: <sip:{phone_number}@{self.gate_ip}>
Call-ID: {session_id}
CSeq: {new_cseq} INVITE
Contact: <sip:{username}@{self.local_ip}:{self.sip_port}>
User-Agent: WindowsVoiceAgent/1.0
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO
Supported: replaces, timer
Authorization: {auth_header}
Content-Type: application/sdp
Content-Length: {len(sdp_content)}

{sdp_content}"""
            
            # Send authenticated INVITE
            self.socket.sendto(authenticated_invite.encode(), (self.gate_ip, self.sip_port))
            logger.info(f"🔑 Sent authenticated INVITE for outbound call to {phone_number}")
            logger.debug(f"Authorization header: {auth_header}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create authenticated INVITE: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _initialize_gemini_early(self, call_id: str):
        """Initialize Gemini connection early (during ringing phase) to reduce post-answer delay"""
        try:
            if call_id not in self.pending_invites:
                logger.warning(f"⚠️ Call ID {call_id} not found in pending invites")
                return
            
            invite_info = self.pending_invites[call_id]
            phone_number = invite_info['phone_number']
            original_number = invite_info.get('original_number', phone_number)
            session_id = invite_info['session_id']
            owner_id = invite_info.get('owner_id')
            custom_config = invite_info.get('custom_config', None)
            
            logger.info(f"🎧 Initializing Gemini connection for {phone_number} (during ringing)...")
            
            # Detect caller's country and language
            caller_country = detect_caller_country(original_number)
            language_info = get_language_config(caller_country)
            voice_config = create_voice_config(language_info, custom_config)
            
            # Get the user's API key for this call
            user_api_key = None
            if owner_id:
                try:
                    from crm_database_mongodb import UserManager
                    db = get_session()
                    user_manager = UserManager(db)
                    user = user_manager.get_user_by_id(owner_id)
                    if user and user.google_api_key:
                        user_api_key = user.google_api_key
                        logger.info(f"🔑 Retrieved API key for user {user.username} (ID: {owner_id})")
                    else:
                        logger.warning(f"⚠️ No API key found for user ID {owner_id}, using default")
                    db.close()
                except Exception as e:
                    logger.error(f"❌ Error retrieving user API key: {e}")
            
            # Create voice session
            voice_session = WindowsVoiceSession(
                session_id, 
                self.phone_number,  # We are calling
                phone_number,       # They are receiving
                voice_config,
                custom_config=custom_config,
                api_key=user_api_key
            )
            
            # Store voice session in pending invites for later use
            self.pending_invites[call_id]['voice_session'] = voice_session
            self.pending_invites[call_id]['language_info'] = language_info
            
            # Create RTP session
            remote_addr = (self.gate_ip, 5004)
            rtp_session = self.rtp_server.create_session(session_id, remote_addr, voice_session, voice_session.call_recorder)
            self.pending_invites[call_id]['rtp_session'] = rtp_session
            
            # Start RTP async loop to initialize Gemini
            if rtp_session and not rtp_session.asyncio_thread_started:
                rtp_session.input_processing = True
                rtp_session.asyncio_thread_started = True
                rtp_session.asyncio_thread = threading.Thread(target=rtp_session._run_asyncio_thread, daemon=True)
                rtp_session.asyncio_thread.start()
                logger.info(f"🎧 Started RTP async loop (Gemini initializing)...")
            
            # Wait for Gemini to connect (with timeout)
            logger.info(f"⏳ Waiting for Gemini to connect (during ringing)...")
            max_wait_time = 30
            wait_interval = 0.5
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                if call_id not in self.pending_invites:
                    logger.warning("⚠️ Call cancelled while initializing Gemini")
                    return
                
                if voice_session.gemini_session:
                    logger.info(f"✅ Gemini connected in {elapsed_time:.1f}s (ready for when user picks up!)")
                    self.pending_invites[call_id]['gemini_ready'] = True
                    break
                
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            
            if not voice_session.gemini_session:
                logger.error(f"❌ Gemini failed to connect within {max_wait_time}s")
                self.pending_invites[call_id]['gemini_failed'] = True
            
        except Exception as e:
            logger.error(f"❌ Error initializing Gemini early: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            if call_id in self.pending_invites:
                self.pending_invites[call_id]['gemini_failed'] = True
    
    def _handle_outbound_call_success(self, message: str):
        """Handle successful outbound call establishment (200 OK for INVITE)"""
        try:
            # Extract Asterisk headers first
            asterisk_headers = self._extract_asterisk_headers(message)
            
            # Extract Call-ID to find the pending invite
            call_id = None
            to_tag = None
            
            for line in message.split('\n'):
                line = line.strip()
                if line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
                elif line.startswith('To:') and 'tag=' in line:
                    tag_start = line.find('tag=') + 4
                    tag_end = line.find(';', tag_start)
                    if tag_end == -1:
                        tag_end = line.find('>', tag_start)
                    if tag_end == -1:
                        tag_end = len(line)
                    to_tag = line[tag_start:tag_end].strip()
            
            if not call_id:
                logger.warning("❌ No Call-ID found in 200 OK response")
                return
            
            # Check if this is a retransmission (call already established)
            if call_id not in self.pending_invites:
                # This is likely a SIP retransmission of 200 OK (normal SIP behavior)
                # Check if session is already active
                for session_id, session_info in active_sessions.items():
                    if session_info.get('call_id') == call_id:
                        # Re-send ACK to acknowledge the retransmission
                        logger.debug(f"📞 Received 200 OK retransmission for call {call_id}, re-sending ACK")
                        # Extract to_tag from the retransmitted message
                        if to_tag and session_info.get('from_tag'):
                            from_tag = session_info.get('from_tag')
                            # Create a minimal invite_info for ACK with all required fields
                            invite_info_for_ack = {
                                'username': config.get('username', '200'),
                                'phone_number': session_info.get('called_number', ''),
                                'session_id': session_id,
                                'cseq': 1  # Use a simple CSeq for retransmission ACK
                            }
                            self._send_ack(call_id, to_tag, from_tag, invite_info_for_ack)
                        return
                # If not in active sessions, it's truly an unknown response
                logger.debug(f"📞 Received 200 OK for unknown call (likely already ended): {call_id}")
                return
            
            invite_info = self.pending_invites[call_id]
            phone_number = invite_info['phone_number']
            original_number = invite_info.get('original_number', phone_number)  # Get original number without prefix
            session_id = invite_info['session_id']
            from_tag = invite_info['from_tag']  # Use our stored from_tag, not from the 200 OK
            was_ringing = invite_info.get('is_ringing', False)  # Check if we saw 180/183 before 200 OK
            owner_id = invite_info.get('owner_id')  # Get owner_id for API key retrieval
            
            # Get linkedid from pending_invites (stored from earlier responses) or from 200 OK
            asterisk_linkedid = invite_info.get('asterisk_linkedid') or asterisk_headers.get('linkedid')
            
            if was_ringing:
                logger.info(f"✅ Outbound call to {phone_number} answered!")
            else:
                logger.warning(f"⚠️ Outbound call to {phone_number} got 200 OK without ringing state (180/183)")
                logger.info(f"📞 Call may still be connecting - will wait for RTP activity before playing greeting")
            
            # Print Asterisk linkedid (the most important place - when call is answered)
            if asterisk_linkedid:
                logger.info(f"🔗 Asterisk Linked ID (Call-Wide): {asterisk_linkedid}")
            else:
                logger.warning("⚠️ No Asterisk Linked ID found in SIP headers")
                logger.warning("💡 Trying to get linkedid from AMI...")
                
                # Try to get linkedid from AMI
                time.sleep(0.2)  # Small delay to let AMI event arrive
                asterisk_linkedid = get_current_linkedid()
                
                if asterisk_linkedid:
                    logger.info(f"🔗 Asterisk Linked ID (from AMI): {asterisk_linkedid}")
                else:
                    logger.warning("⚠️ No linkedid available from AMI either")
                    logger.warning("💡 Check AMI connection and Asterisk configuration")
            
            # Detect caller's country and language (for outbound calls, use target number)
            caller_country = detect_caller_country(original_number)  # Use original number for language
            language_info = get_language_config(caller_country)
            
            # Get custom config if available from pending invite
            custom_config = invite_info.get('custom_config', None)
            
            voice_config = create_voice_config(language_info, custom_config)
            
            logger.info(f"🌍 Language: {language_info['lang']} ({caller_country})")
            
            # 🚀 NEW: Check if Gemini was already initialized during ringing
            gemini_ready = invite_info.get('gemini_ready', False)
            voice_session = invite_info.get('voice_session', None)
            rtp_session = invite_info.get('rtp_session', None)
            
            if gemini_ready and voice_session and rtp_session:
                logger.info("✅ Reusing Gemini connection initialized during ringing (FAST PATH)")
                
                # ✅ PROFESSIONAL: Use state machine method to handle call answered
                rtp_session.on_call_answered()
                logger.info("📞 Call answered - state machine updated, AMD detection started")
                
                # Store SIP details in voice session for sending BYE later
                voice_session.sip_handler = self
                voice_session.from_tag = from_tag
                voice_session.to_tag = to_tag
                voice_session.call_id = call_id
            else:
                # FALLBACK: Old behavior - initialize Gemini after answer
                if gemini_ready:
                    logger.warning("⚠️ Gemini was ready but sessions missing, recreating...")
                else:
                    logger.warning("⚠️ Gemini not initialized during ringing, initializing now (SLOW PATH)")
                
                # Get the user's API key for this call
                user_api_key = None
                if owner_id:
                    try:
                        from crm_database_mongodb import UserManager
                        db = get_session()
                        user_manager = UserManager(db)
                        user = user_manager.get_user_by_id(owner_id)
                        if user and user.google_api_key:
                            user_api_key = user.google_api_key
                            logger.info(f"🔑 Retrieved API key for user {user.username} (ID: {owner_id})")
                        else:
                            logger.warning(f"⚠️ No API key found for user ID {owner_id}, using default")
                        db.close()
                    except Exception as e:
                        logger.error(f"❌ Error retrieving user API key: {e}")
                else:
                    logger.warning(f"⚠️ No owner_id for outbound call, using default API key")
                
                # Create voice session for outbound call (we are the caller)
                voice_session = WindowsVoiceSession(
                    session_id, 
                    self.phone_number,  # We are calling
                    phone_number,       # They are receiving
                    voice_config,
                    custom_config=custom_config,
                    api_key=user_api_key
                )
                
                # Store SIP details in voice session for sending BYE later
                voice_session.sip_handler = self
                voice_session.from_tag = from_tag
                voice_session.to_tag = to_tag
                voice_session.call_id = call_id
                
                # ⚠️ CRITICAL FIX: Check if RTP session already exists before creating a new one
                # This prevents orphaned RTP sessions that cause cross-session contamination
                if session_id in self.rtp_server.sessions:
                    logger.warning(f"⚠️ RTP session {session_id} already exists! Cleaning up old session before creating new one")
                    # Clean up the old session first to prevent orphaned threads
                    self.rtp_server.remove_session(session_id)
                
                # Create RTP session - for outbound calls, we need to get the remote address from SDP
                # For now, we'll use the Gate VoIP address as the RTP destination
                remote_addr = (self.gate_ip, 5004)  # Default RTP port
                rtp_session = self.rtp_server.create_session(session_id, remote_addr, voice_session, voice_session.call_recorder)
                
                # ✅ PROFESSIONAL: Use state machine method to handle call answered
                rtp_session.on_call_answered()
                logger.info("📞 Call answered (slow path) - state machine updated, AMD detection started")
            
            # Store the active session
            active_sessions[session_id] = {
                "voice_session": voice_session,
                "rtp_session": rtp_session,
                "caller_addr": (self.gate_ip, self.sip_port),
                "status": "connecting",
                "call_start": datetime.now(timezone.utc),
                "call_id": call_id,
                "call_type": "outbound",
                "sip_handler": self,
                "from_tag": from_tag,
                "to_tag": to_tag,
                "caller_id": self.phone_number,
                "called_number": phone_number,
                "was_ringing": was_ringing,  # Track if we saw 180/183 before 200 OK
                "asterisk_linkedid": asterisk_linkedid  # Store Asterisk Linked ID for the call
            }
            
            # Send ACK to complete the call setup BEFORE initializing Gemini
            self._send_ack(call_id, to_tag, from_tag, invite_info)
            logger.info(f"✅ Outbound call {session_id} established")
            
            # Update CallSession in database with asterisk_linkedid (for outbound calls only)
            if asterisk_linkedid:
                try:
                    db = get_session()
                    try:
                        crm_session = db.query(CallSession).filter(CallSession.session_id == session_id).first()
                        if crm_session:
                            crm_session.asterisk_linkedid = asterisk_linkedid
                            crm_session.status = CallStatus.ANSWERED
                            crm_session.answered_at = datetime.utcnow()
                            crm_session.save()  # MongoDB requires explicit save
                            logger.info(f"💾 Saved Asterisk Linked ID to database: {asterisk_linkedid}")
                        else:
                            logger.warning(f"⚠️ CallSession not found for session_id: {session_id}")
                    except Exception as e:
                        logger.error(f"Error updating CallSession with linkedid: {e}")
                    finally:
                        db.close()
                except Exception as e:
                    logger.error(f"Error in database update: {e}")
            else:
                logger.warning("⚠️ No Asterisk Linked ID available to save")
            
            # Clean up pending invite
            del self.pending_invites[call_id]
            
            # NOW start voice session after ACK is sent
            def start_voice_session():
                try:
                    # Check if Gemini was already initialized during ringing
                    if gemini_ready and voice_session.gemini_session:
                        logger.info("🚀 Gemini already connected (fast path - no wait needed!)")
                    else:
                        # Start the RTP session's async processing loop (which will initialize Gemini)
                        if rtp_session and not rtp_session.asyncio_thread_started:
                            rtp_session.input_processing = True
                            rtp_session.asyncio_thread_started = True  # Set flag before starting thread
                            rtp_session.asyncio_thread = threading.Thread(target=rtp_session._run_asyncio_thread, daemon=True)
                            rtp_session.asyncio_thread.start()
                            logger.info(f"🎧 Started RTP async loop for outbound session {session_id}")
                        elif rtp_session and rtp_session.asyncio_thread_started:
                            logger.info(f"🎧 RTP async loop already started for outbound session {session_id}")
                        
                        # Wait for Gemini to connect (check every 0.5s, timeout after 30s)
                        logger.info(f"⏳ Waiting for Gemini to connect for outbound call...")
                        max_wait_time = 30  # seconds
                        wait_interval = 0.5  # seconds
                        elapsed_time = 0
                        
                        while elapsed_time < max_wait_time:
                            if session_id not in active_sessions:
                                logger.warning("⚠️ Outbound call ended while waiting for Gemini")
                                return
                            
                            # Check if Gemini session is ready
                            if voice_session.gemini_session:
                                logger.info(f"✅ Gemini connected for outbound call after {elapsed_time:.1f}s")
                                break
                            
                            time.sleep(wait_interval)
                            elapsed_time += wait_interval
                        
                        if not voice_session.gemini_session:
                            logger.error(f"❌ Gemini failed to connect for outbound call within {max_wait_time}s")
                            if session_id in active_sessions:
                                del active_sessions[session_id]
                            return

                    # Mark as active once voice session is ready
                    if session_id in active_sessions:
                        active_sessions[session_id]["status"] = "active"
                        logger.info(f"🎯 Outbound voice session {session_id} is now active and ready")

                        # Play greeting now that call is fully established AND Gemini is ready
                        def play_outbound_greeting():
                            # ✅ Skip file-based greeting if gemini_greeting is enabled
                            if custom_config and custom_config.get('gemini_greeting', False):
                                logger.info("🎙️ Gemini greeting enabled - skipping file-based greeting (Gemini will speak first)")
                                return
                            
                            # Check if we saw ringing state before 200 OK
                            was_ringing = active_sessions[session_id].get('was_ringing', False) if session_id in active_sessions else False
                            
                            if was_ringing:
                                # Normal case: We saw 180/183 ringing, so 200 OK means user picked up
                                # 🚀 NEW: Wait only 1.0 second (reduced from 1.5s)
                                time.sleep(1.0)
                            else:
                                # Special case: No 180/183 ringing seen, 200 OK may be premature
                                # Wait for actual RTP activity to confirm user picked up
                                logger.info("⏳ No ringing state detected - waiting for RTP activity before playing greeting")
                                max_wait = 15  # Wait up to 15 seconds for RTP activity
                                wait_interval = 0.1
                                elapsed = 0
                                
                                while elapsed < max_wait:
                                    if session_id not in active_sessions:
                                        logger.warning("⚠️ Call ended while waiting for RTP activity")
                                        return
                                    
                                    # Check if we've received any RTP packets (indicates phone was picked up)
                                    rtp_sess = active_sessions[session_id].get('rtp_session')
                                    if rtp_sess and hasattr(rtp_sess, 'packets_received') and rtp_sess.packets_received > 5:
                                        logger.info(f"✅ RTP activity detected after {elapsed:.1f}s - user has picked up")
                                        # 🚀 NEW: Wait only 1.0 second (reduced from 1.5s)
                                        time.sleep(1.0)
                                        break
                                    
                                    time.sleep(wait_interval)
                                    elapsed += wait_interval
                                
                                if elapsed >= max_wait:
                                    logger.warning("⚠️ No RTP activity detected within 15s, playing greeting anyway")
                            
                            # Check if call is still active
                            if session_id not in active_sessions:
                                logger.warning("⚠️ Outbound call ended before greeting could be played")
                                return
                            
                            logger.info("🎵 Playing greeting to called party...")
                            
                            # Check if we need to generate greeting from call_config
                            custom_greeting = None
                            if custom_config:
                                # If greeting_file is already provided, use it
                                if custom_config.get('greeting_file'):
                                    custom_greeting = custom_config.get('greeting_file')
                                # Otherwise, if call_config is provided, generate greeting now
                                elif custom_config.get('call_config') and custom_config.get('phone_number'):
                                    try:
                                        logger.info("🎤 Generating greeting from call_config...")
                                        phone_num = custom_config.get('phone_number')
                                        call_cfg = custom_config.get('call_config')
                                        
                                        # Detect language for greeting
                                        caller_ctry = detect_caller_country(phone_num)
                                        lang_info = get_language_config(caller_ctry)
                                        
                                        # Generate greeting using Gemini
                                        import asyncio
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        greeting_result = loop.run_until_complete(
                                            generate_greeting_for_lead(
                                                language=lang_info['lang'],
                                                language_code=lang_info['code'],
                                                call_config=call_cfg
                                            )
                                        )
                                        loop.close()
                                        
                                        if greeting_result and greeting_result.get('success'):
                                            custom_greeting = greeting_result.get('greeting_file')
                                            logger.info(f"✅ Generated greeting: {custom_greeting}")
                                        else:
                                            logger.warning("⚠️ Failed to generate greeting")
                                    except Exception as e:
                                        logger.warning(f"⚠️ Error generating greeting: {e}")
                            
                            if not custom_greeting:
                                custom_greeting = 'greeting.wav'
                            
                            # Only play greeting if file exists
                            greeting_duration = 0
                            if custom_greeting and os.path.exists(custom_greeting):
                                greeting_duration = rtp_session.play_greeting_file(custom_greeting)
                            else:
                                logger.warning(f"⚠️ Greeting file not found or not specified: {custom_greeting}")
                            
                            if greeting_duration > 0:
                                logger.info(f"✅ Greeting played ({greeting_duration:.1f}s). Ready for conversation.")
                            else:
                                logger.warning("⚠️ Greeting file not found or failed to play")
                        
                        threading.Thread(target=play_outbound_greeting, daemon=True).start()
                        
                except Exception as e:
                    logger.error(f"❌ Failed to start outbound voice session: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Remove failed session
                    if session_id in active_sessions:
                        del active_sessions[session_id]
            
            threading.Thread(target=start_voice_session, daemon=True).start()
            
        except Exception as e:
            logger.error(f"❌ Error handling outbound call success: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _send_ack(self, call_id: str, to_tag: str, from_tag: str, invite_info: dict):
        """Send ACK to complete call establishment"""
        try:
            username = invite_info['username']
            phone_number = invite_info['phone_number']
            session_id = invite_info['session_id']
            cseq = invite_info['cseq']
            
            ack_message = f"""ACK sip:{phone_number}@{self.gate_ip} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{session_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={from_tag}
To: <sip:{phone_number}@{self.gate_ip}>;tag={to_tag}
Call-ID: {call_id}
CSeq: {cseq} ACK
User-Agent: WindowsVoiceAgent/1.0
Content-Length: 0

"""
            
            self.socket.sendto(ack_message.encode(), (self.gate_ip, self.sip_port))
            
        except Exception as e:
            logger.error(f"❌ Error sending ACK: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _extract_asterisk_headers(self, message: str) -> dict:
        """Extract Asterisk-specific headers from SIP message
        
        Returns:
            dict with keys: linkedid, uniqueid (if present in headers)
        """
        asterisk_info = {}
        for line in message.split('\n'):
            line = line.strip()
            # Check for X-Asterisk headers (case-insensitive)
            if line.lower().startswith('x-asterisk-linkedid:'):
                asterisk_info['linkedid'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('x-asterisk-uniqueid:'):
                asterisk_info['uniqueid'] = line.split(':', 1)[1].strip()
            # Some Asterisk versions may use different header names
            elif line.lower().startswith('x-linkedid:'):
                asterisk_info['linkedid'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('x-uniqueid:'):
                asterisk_info['uniqueid'] = line.split(':', 1)[1].strip()
        return asterisk_info
    
    def _handle_sip_response(self, message: str, addr):
        """Handle SIP responses (100 Trying, 180 Ringing, 183 Session Progress, 200 OK, 401 Unauthorized, etc.)"""
        try:
            first_line = message.split('\n')[0].strip()
            
            # Extract Asterisk headers if present
            asterisk_headers = self._extract_asterisk_headers(message)
            
            if '100 Trying' in first_line:
                # Handle 100 Trying response for outbound calls
                if 'INVITE' in message or 'CSeq:' in message:
                    # Extract Call-ID to find the pending invite
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        phone_number = self.pending_invites[call_id]['phone_number']
                        logger.info(f"📞 Call initiated - processing call to {phone_number}")
                        # Print Asterisk linkedid if available
                        if asterisk_headers.get('linkedid'):
                            logger.info(f"🔗 Asterisk Linked ID: {asterisk_headers['linkedid']}")
                            # Store it in pending_invites for later use
                            self.pending_invites[call_id]['asterisk_linkedid'] = asterisk_headers['linkedid']
                    else:
                        logger.debug("📞 Received 100 Trying response")
                else:
                    logger.debug(f"Received 100 Trying response: {first_line}")
            
            elif '180 Ringing' in first_line:
                # Handle 180 Ringing response for outbound calls
                if 'INVITE' in message or 'CSeq:' in message:
                    # Extract Call-ID to find the pending invite
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        # Mark that we've seen ringing state
                        self.pending_invites[call_id]['is_ringing'] = True
                        phone_number = self.pending_invites[call_id]['phone_number']
                        logger.info(f"🔔 Phone ringing - outbound call to {phone_number}")
                        # Print Asterisk linkedid if available
                        if asterisk_headers.get('linkedid'):
                            logger.info(f"🔗 Asterisk Linked ID: {asterisk_headers['linkedid']}")
                            # Store it in pending_invites for later use
                            self.pending_invites[call_id]['asterisk_linkedid'] = asterisk_headers['linkedid']
                        
                        # 🚀 NEW: Initialize Gemini connection early (during ringing, not after answer)
                        # This reduces the 6-7s delay before greeting playback
                        if not self.pending_invites[call_id].get('gemini_initializing'):
                            self.pending_invites[call_id]['gemini_initializing'] = True
                            logger.info("🎧 Starting Gemini connection early (during ringing)...")
                            threading.Thread(target=self._initialize_gemini_early, args=(call_id,), daemon=True).start()
                    else:
                        logger.info("🔔 Phone ringing - waiting for answer")
                else:
                    logger.debug(f"Received 180 Ringing response: {first_line}")
            
            elif '183 Session Progress' in first_line:
                # Handle 183 Session Progress response (ringing with early media)
                if 'INVITE' in message or 'CSeq:' in message:
                    # Extract Call-ID to find the pending invite
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        # Mark that we've seen ringing state (183 = ringing with early media)
                        self.pending_invites[call_id]['is_ringing'] = True
                        phone_number = self.pending_invites[call_id]['phone_number']
                        logger.info(f"🔔 Phone ringing with early media - outbound call to {phone_number}")
                        # Print Asterisk linkedid if available
                        if asterisk_headers.get('linkedid'):
                            logger.info(f"🔗 Asterisk Linked ID: {asterisk_headers['linkedid']}")
                            # Store it in pending_invites for later use
                            self.pending_invites[call_id]['asterisk_linkedid'] = asterisk_headers['linkedid']
                        
                        # 🚀 NEW: Initialize Gemini connection early (during ringing, not after answer)
                        # This reduces the 6-7s delay before greeting playback
                        if not self.pending_invites[call_id].get('gemini_initializing'):
                            self.pending_invites[call_id]['gemini_initializing'] = True
                            logger.info("🎧 Starting Gemini connection early (during ringing)...")
                            threading.Thread(target=self._initialize_gemini_early, args=(call_id,), daemon=True).start()
                    else:
                        logger.info("🔔 Phone ringing with early media - waiting for answer")
                else:
                    logger.debug(f"Received 183 Session Progress response: {first_line}")
            
            elif '200 OK' in first_line:
                if 'REGISTER' in message:
                    # Check if this is extension registration based on Call-ID
                    call_id_line = [line for line in message.split('\n') if line.strip().startswith('Call-ID:')]
                    if call_id_line and self.registration_call_id and self.registration_call_id in call_id_line[0]:
                        logger.info("✅ Successfully registered as Extension 200!")
                        logger.info("📞 Voice agent can now make outbound calls")
                        self.extension_registered = True
                    else:
                        logger.info("✅ Successfully registered with Gate VoIP!")
                        logger.info("🎯 Voice agent is now ready to receive calls")
                elif 'INVITE' in message:
                    # Handle successful outbound call establishment
                    self._handle_outbound_call_success(message)
                else:
                    logger.debug(f"Received 200 OK response: {first_line}")
            
            elif '401 Unauthorized' in first_line:
                if 'REGISTER' in message:
                    # Check if this is for extension registration
                    call_id_line = [line for line in message.split('\n') if line.strip().startswith('Call-ID:')]
                    is_extension_registration = call_id_line and self.registration_call_id and self.registration_call_id in call_id_line[0]
                    
                    if is_extension_registration:
                        logger.info("🔐 Received authentication challenge for Extension 200 registration")
                        # Send authenticated REGISTER for extension
                        self._create_authenticated_extension_register(message)
                    else:
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
                elif 'INVITE' in message:
                    # Handle authentication challenge for outbound INVITE
                    logger.info("🔐 Received authentication challenge for outbound INVITE")
                    self._create_authenticated_invite(message)
                else:
                    logger.warning("🔐 Received 401 for unknown message type")
                
            elif '403 Forbidden' in first_line:
                logger.error(f"❌ Authentication failed: {first_line}")
                logger.error("💡 Check username/password in asterisk_config.json")
                logger.error("💡 Verify extension 200 is properly configured in Gate VoIP")
                
            elif '404 Not Found' in first_line:
                if 'INVITE' in message:
                    # Extract Call-ID to find the call that failed
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        phone_number = self.pending_invites[call_id]['phone_number']
                        session_id = self.pending_invites[call_id]['session_id']
                        logger.error(f"❌ Outbound call to {phone_number} failed: {first_line}")
                        logger.error("💡 This usually means the outgoing route is misconfigured")
                        logger.error("💡 CRITICAL: Check Gate VoIP outgoing route uses GSM2 trunk (not voice-agent trunk)")
                        logger.error("💡 Go to: http://192.168.50.50 > PBX Settings > Outgoing Routes")
                        logger.error("💡 Change trunk from 'voice-agent' to 'gsm2' and save")
                        
                        # Update call session status
                        self._update_call_session_status(session_id, CallStatus.FAILED)
                        
                        # Clean up the failed call
                        del self.pending_invites[call_id]
                    else:
                        logger.error(f"❌ Call not found: {first_line}")
                else:
                    logger.error(f"❌ User not found: {first_line}")
                    logger.error("💡 Check if extension 200 exists in Gate VoIP configuration")
            
            elif '486 Busy Here' in first_line or '600 Busy Everywhere' in first_line:
                # User is busy - can't answer
                if 'INVITE' in message:
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        phone_number = self.pending_invites[call_id]['phone_number']
                        session_id = self.pending_invites[call_id]['session_id']
                        logger.info(f"📞 Call to {phone_number} - user busy")
                        
                        # Update call session status
                        self._update_call_session_status(session_id, CallStatus.REJECTED)
                        
                        # Clean up
                        del self.pending_invites[call_id]
            
            elif '487 Request Terminated' in first_line:
                # Call was cancelled/terminated (usually by caller)
                if 'INVITE' in message:
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        phone_number = self.pending_invites[call_id]['phone_number']
                        session_id = self.pending_invites[call_id]['session_id']
                        logger.info(f"📞 Call to {phone_number} - request terminated")
                        
                        # Update call session status  
                        self._update_call_session_status(session_id, CallStatus.FAILED)
                        
                        # Clean up
                        del self.pending_invites[call_id]
            
            elif '603 Decline' in first_line or '480 Temporarily Unavailable' in first_line:
                # User declined the call or temporarily unavailable
                if 'INVITE' in message:
                    call_id = None
                    for line in message.split('\n'):
                        line = line.strip()
                        if line.startswith('Call-ID:'):
                            call_id = line.split(':', 1)[1].strip()
                            break
                    
                    if call_id and call_id in self.pending_invites:
                        phone_number = self.pending_invites[call_id]['phone_number']
                        session_id = self.pending_invites[call_id]['session_id']
                        logger.info(f"📞 Call to {phone_number} - declined/unavailable")
                        
                        # Update call session status
                        self._update_call_session_status(session_id, CallStatus.REJECTED)
                        
                        # Clean up
                        del self.pending_invites[call_id]
                
            else:
                logger.info(f"📨 SIP Response: {first_line}")
                if '401' in first_line or '403' in first_line or '404' in first_line:
                    logger.debug(f"Full response message:\n{message}")
                
        except Exception as e:
            logger.error(f"Error handling SIP response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_call_session_status(self, session_id: str, status):
        """Update call session status in database"""
        try:
            from crm_database import CallStatus as CRMCallStatus
            db = get_session()
            try:
                crm_session = db.query(CallSession).filter(CallSession.session_id == session_id).first()
                if crm_session:
                    crm_session.status = status
                    crm_session.ended_at = datetime.utcnow()
                    if crm_session.started_at and crm_session.ended_at:
                        crm_session.duration = int((crm_session.ended_at - crm_session.started_at).total_seconds())
                    crm_session.save()  # MongoDB requires explicit save
                    logger.info(f"📋 Updated call session {session_id} status to {status}")
            except Exception as e:
                logger.error(f"Error updating call session status: {e}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error in _update_call_session_status: {e}")
    
    def stop(self):
        """Stop SIP handler"""
        self.running = False
        if self.socket:
            self.socket.close()
        
        # Stop RTP server
        self.rtp_server.stop()

class WindowsVoiceSession:
    """Voice session for Windows - simpler than Linux version"""
    
    def __init__(self, session_id: str, caller_id: str, called_number: str, voice_config: Dict[str, Any] = None, custom_config: Dict[str, Any] = None, api_key: str = None):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        self.session_logger = SessionLogger()
        self.voice_session = None
        self.gemini_session = None  # The actual session object from the context manager
        self._init_lock = threading.Lock()  # Lock to prevent concurrent initialization
        self._initializing = False  # Flag to track if initialization is in progress
        
        # Store API key for this session
        self.api_key = api_key
        
        # Create a session-specific genai client if api_key is provided
        if api_key:
            self.session_voice_client = genai.Client(
                api_key=api_key,
                http_options={"api_version": "v1beta"}
            )
            logger.info(f"🔑 Using user-specific API key for session {session_id}: {api_key[:20]}...{api_key[-4:]}")
        else:
            # Fallback to global client
            self.session_voice_client = voice_client
            logger.warning(f"⚠️ No user-specific API key for session {session_id}, using global client")
        
        # Store custom config for later use
        self.custom_config = custom_config
        
        # SIP details for sending BYE (will be set after call is established)
        self.sip_handler = None
        self.from_tag = None
        self.to_tag = None
        self.call_id = None
        
        # Initialize call recorder
        self.call_recorder = CallRecorder(session_id, caller_id, called_number)
        logger.info(f"🎙️ Call recorder initialized for session {session_id}")
        
        # Use provided voice config or default (will be recreated if custom config is provided)
        self.voice_config = voice_config if voice_config else DEFAULT_VOICE_CONFIG
        
        # Connection management
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.connection_backoff = 1.0  # seconds
        self.last_connection_attempt = 0
        
        # Audio resampling state tuples for audioop.ratecv
        # Format: (last_fragment, combined_state)
        self._resample_in_state = None  
        self._resample_out_state = None
        
        # Log call start with language info
        system_text = ''
        if hasattr(self.voice_config, 'system_instruction') and self.voice_config.system_instruction:
            if hasattr(self.voice_config.system_instruction, 'parts') and self.voice_config.system_instruction.parts:
                system_text = self.voice_config.system_instruction.parts[0].text if self.voice_config.system_instruction.parts else ''
        
        if 'АртроФлекс' in system_text or 'болки в ставите' in system_text:
            detected_lang = 'Bulgarian'
        elif 'ArtroFlex' in system_text and 'joint pain' in system_text:
            detected_lang = 'English'
        elif 'Romanian' in system_text:
            detected_lang = 'Romanian'
        elif 'Greek' in system_text:
            detected_lang = 'Greek'
        else:
            detected_lang = 'Unknown'
        
        self.session_logger.log_transcript(
            "system", 
            f"Call started - From: {caller_id}, To: {called_number}, Language: {detected_lang}"
        )
    
    async def initialize_voice_session(self):
        """Initialize the Google Gemini voice session with improved error handling"""
        # Check if session is already initialized or being initialized
        with self._init_lock:
            if self.gemini_session is not None:
                logger.warning(f"⚠️ Voice session already initialized for {self.session_id}, skipping duplicate initialization")
                return True
            
            if self._initializing:
                logger.warning(f"⚠️ Voice session initialization already in progress for {self.session_id}, skipping duplicate call")
                return False
            
            # Mark that we're initializing
            self._initializing = True
        
        try:
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
            
            logger.info(f"Attempting to connect to Gemini live session (attempt {self.connection_attempts}/{self.max_connection_attempts})...")
            logger.info(f"Using model: {MODEL}")
            # Extract language from voice config for logging
            system_text = ''
            if hasattr(self.voice_config, 'system_instruction') and self.voice_config.system_instruction:
                if hasattr(self.voice_config.system_instruction, 'parts') and self.voice_config.system_instruction.parts:
                    system_text = self.voice_config.system_instruction.parts[0].text if self.voice_config.system_instruction.parts else ''
            
            if 'АртроФлекс' in system_text or 'болки в ставите' in system_text:
                detected_lang = 'Bulgarian'
            elif 'ArtroFlex' in system_text and 'joint pain' in system_text:
                detected_lang = 'English'
            elif 'Romanian' in system_text:
                detected_lang = 'Romanian'
            elif 'Greek' in system_text:
                detected_lang = 'Greek'
            else:
                detected_lang = 'Unknown'
            logger.info(f"Voice config language: {detected_lang}")
            
            # Create the connection with timeout
            try:
                # Create the context manager with the dynamic voice config
                # Use session-specific client to ensure isolation between concurrent calls
                context_manager = self.session_voice_client.aio.live.connect(
                    model=MODEL, 
                    config=self.voice_config
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
            
            # Connection is ready immediately - no delay needed
            # The AI will respond naturally when it receives actual caller audio
            logger.info(f"🎤 Voice session ready - waiting for caller audio...")
            
            logger.info(f"🎤 Waiting for caller audio to trigger AI responses...")
            
            # Log successful initialization
            self.session_logger.log_transcript(
                "system", 
                f"Voice session initialized - ready for conversation"
            )
            
            # Reset initializing flag on success
            with self._init_lock:
                self._initializing = False
            
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
            
            # Reset initializing flag on failure
            with self._init_lock:
                self._initializing = False
            
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
                        input={"data": processed_audio, "mime_type": "audio/pcm;rate=24000"},
                        end_of_turn=False
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
            
            # Get response immediately without timeout - continuous streaming
            response_audio = b""
            
            try:
                # Receive response immediately without any timeout delays
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
        """
        Input: 16-bit mono PCM at 8000 Hz.
        Output: 16-bit mono PCM at 24000 Hz.
        Using audioop.ratecv (C-implementation) for minimum latency.
        """
        try:
            # ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0)
            # We use weightA=1, weightB=0 for simple linear interpolation (fastest)
            new_fragment, self._resample_in_state = audioop.ratecv(
                audio_data, 
                2, # 16-bit
                1, # Mono
                8000, 
                24000, 
                self._resample_in_state
            )
            return new_fragment
        except Exception as e:
            logger.warning(f"audioop upsample failed: {e}")
            return audio_data # Fail safe (audio will sound slow/deep but stream continues)

    def convert_gemini_to_telephony(self, model_pcm: bytes) -> bytes:
        """
        Input: 16-bit mono PCM at 24000 Hz.
        Output: 16-bit mono PCM at 8000 Hz.
        Using audioop.ratecv for minimum latency.
        """
        try:
            # Downsample 24k -> 8k
            new_fragment, self._resample_out_state = audioop.ratecv(
                model_pcm, 
                2, # 16-bit
                1, # Mono
                24000, 
                8000, 
                self._resample_out_state
            )
            return new_fragment
        except Exception as e:
            logger.warning(f"audioop downsample failed: {e}")
            return model_pcm # Fail safe
    
    
    def cleanup(self):
        """Clean up the session synchronously"""
        try:
            # Stop call recording
            if hasattr(self, 'call_recorder') and self.call_recorder:
                self.call_recorder.stop_recording()
                logger.info(f"📼 Recording stopped for session {self.session_id}")
            
            # Try to close the voice session properly if it exists
            if self.voice_session and self.gemini_session:
                try:
                    # Create a new event loop just for cleanup if needed
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, just clear the sessions
                            logger.info("Event loop running, clearing sessions without async cleanup")
                            self.voice_session = None
                            self.gemini_session = None
                        else:
                            # Run the async cleanup
                            loop.run_until_complete(self._async_close_session())
                    except RuntimeError:
                        # No event loop, create one for cleanup
                        asyncio.run(self._async_close_session())
                except Exception as cleanup_error:
                    logger.warning(f"Error in async cleanup, falling back to simple cleanup: {cleanup_error}")
                    self.voice_session = None
                    self.gemini_session = None
            else:
                # Simply clear the sessions
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
            # Stop call recording
            if hasattr(self, 'call_recorder') and self.call_recorder:
                self.call_recorder.stop_recording()
                logger.info(f"📼 Recording stopped for session {self.session_id}")
            
            await self._async_close_session()
            
            self.session_logger.log_transcript("system", "Call ended")
            self.session_logger.save_session()
            self.session_logger.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

# Initialize SIP handler (must be after app definition to avoid circular reference issues)
sip_handler = WindowsSIPHandler()

@app.post("/api/generate_greeting")
async def generate_greeting(greeting_request: dict):
    """Generate custom greeting audio file for a lead with specified voice and greeting text"""
    try:
        if not GREETING_GENERATOR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Greeting generator not available - Gemini API not configured"
            )
        
        phone_number = greeting_request.get("phone_number")
        call_config = greeting_request.get("call_config", {})
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="Phone number required")
        
        # Detect language from phone number
        logger.info(f"🎤 Generating greeting for {phone_number}")
        caller_country = detect_caller_country(phone_number)
        language_info = get_language_config(caller_country)
        
        # Extract voice and greeting instruction from call_config
        voice_name = call_config.get("voice_name", "Puck")
        greeting_instruction = call_config.get("greeting_instruction")
        logger.info(f"🌍 Detected language: {language_info['lang']} ({language_info['code']})")
        logger.info(f"🎙️ Using voice: {voice_name}")
        if greeting_instruction:
            logger.info(f"📝 Custom greeting instruction: {greeting_instruction[:100]}...")
        
        # Generate greeting
        result = await generate_greeting_for_lead(
            language=language_info['lang'],
            language_code=language_info['code'],
            call_config=call_config
        )
        
        if result.get("success"):
            return {
                "status": "success",
                "greeting_file": result["greeting_file"],
                "transcript": result["transcript"],
                "language": result["language"],
                "language_code": result["language_code"],
                "voice": result.get("voice", voice_name),
                "greeting_text": result["transcript"]  # ✅ Include greeting text for context
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to generate greeting")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/make_call")
async def make_outbound_call(call_request: dict, current_user: User = Depends(check_subscription)):
    """API endpoint to initiate an outbound call with optional custom prompt"""
    try:
        phone_number = call_request.get("phone_number")
        call_config = call_request.get("call_config", {})
        greeting_file = call_request.get("greeting_file")  # Optional custom greeting
        greeting_transcript = call_request.get("greeting_transcript")  # ✅ Greeting text for context
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="Phone number required")
        
        # Extract custom prompt configuration
        custom_config = {
            "company_name": call_config.get("company_name", "PropTechAI"),
            "caller_name": call_config.get("caller_name", "Assistant"),
            "product_name": call_config.get("product_name", "our product"),
            "additional_prompt": call_config.get("additional_prompt", ""),
            "call_urgency": call_config.get("call_urgency", "medium"),
            "call_objective": call_config.get("call_objective", "sales"),
            "main_benefits": call_config.get("main_benefits", ""),
            "special_offer": call_config.get("special_offer", ""),
            "objection_strategy": call_config.get("objection_strategy", "understanding"),
            "voice_name": call_config.get("voice_name", "Puck"),  # ✅ Pass voice from CRM to custom_config
            "gemini_greeting": call_config.get("gemini_greeting", False),  # ✅ Let Gemini speak the greeting
            "greeting_instruction": call_config.get("greeting_instruction", "")  # ✅ Custom greeting text for Gemini
        }
        
        # 🧪 TEMPORARY BYPASS FOR TESTING: Force gemini_greeting=True to test the feature
        # Comment out this block after testing is complete
        if True:  # Set to False to disable bypass
            custom_config['gemini_greeting'] = True
            logger.info("🧪 TEST MODE: gemini_greeting FORCED ON - Gemini will speak first!")
        
        logger.info(f"📞 Making outbound call to {phone_number} with custom config:")
        logger.info(f"   Company: {custom_config['company_name']}")
        logger.info(f"   Caller: {custom_config['caller_name']}")
        logger.info(f"   Product: {custom_config['product_name']}")
        logger.info(f"   Objective: {custom_config['call_objective']}")
        logger.info(f"   Urgency: {custom_config['call_urgency']}")
        logger.info(f"   Benefits: {custom_config['main_benefits']}")
        logger.info(f"   Offers: {custom_config['special_offer']}")
        logger.info(f"   Objection Strategy: {custom_config['objection_strategy']}")
        logger.info(f"   Voice: {custom_config['voice_name']}")  # ✅ Log selected voice
        
        # Store greeting file in custom config if provided (skip if gemini_greeting is enabled)
        if greeting_file and not custom_config.get('gemini_greeting', False):
            custom_config['greeting_file'] = greeting_file
            logger.info(f"🎵 Using custom greeting: {greeting_file}")
        elif custom_config.get('gemini_greeting', False):
            logger.info(f"🎙️ Gemini greeting mode - ignoring pre-generated greeting file")
        
        # ✅ Store greeting transcript for context (skip if gemini_greeting - Gemini will create its own)
        if greeting_transcript and not custom_config.get('gemini_greeting', False):
            custom_config['greeting_transcript'] = greeting_transcript
            logger.info(f"📝 Greeting transcript: {greeting_transcript[:100]}...")
        
        # Get the agent's gate slot
        gate_slot = None
        if current_user and current_user.gate_slot:
            gate_slot = current_user.gate_slot
            logger.info(f"📞 Using gate slot {gate_slot} for agent {current_user.username}")
        else:
            logger.warning(f"⚠️ User has no assigned gate slot, will use default")
        
        # Make call through Gate VoIP with custom config and agent's gate slot
        session_id = sip_handler.make_outbound_call(
            phone_number, 
            custom_config, 
            gate_slot=gate_slot,
            owner_id=current_user.id if current_user else None
        )
        
        if session_id:
            # Save to CRM database
            db = get_session()
            try:
                # Find lead by phone number - need to search all leads and match using full_phone property
                # since phone and country_code are stored separately
                lead = None
                
                # Normalize phone number for comparison (remove spaces, dashes, etc.)
                normalized_phone = phone_number.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
                
                # Try exact match on phone field first (for numbers stored without country code)
                lead = db.query(Lead).filter(Lead.phone == phone_number).first()
                
                if not lead:
                    # Try without + prefix
                    lead = db.query(Lead).filter(Lead.phone == phone_number.replace('+', '')).first()
                
                if not lead:
                    # Search for leads where the stored phone matches our number
                    # Get last 10 digits of the phone number for comparison
                    last_digits = normalized_phone[-10:] if len(normalized_phone) >= 10 else normalized_phone
                    
                    # Get all leads and check full_phone property
                    all_leads = db.query(Lead).all()
                    for potential_lead in all_leads:
                        if potential_lead.full_phone:
                            lead_normalized = potential_lead.full_phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
                            # Check if phone numbers match (exact or by last 10 digits)
                            if lead_normalized == normalized_phone or (len(lead_normalized) >= 10 and lead_normalized[-10:] == last_digits):
                                lead = potential_lead
                                break
                
                # Update lead if found
                if lead:
                    lead.call_count += 1
                    lead.last_called_at = datetime.utcnow()
                    db.add(lead)  # Explicitly add to session for MongoDB
                    logger.info(f"✅ Updated lead {lead.id} ({lead.full_name}): call_count={lead.call_count}, last_called_at={lead.last_called_at}")
                else:
                    logger.warning(f"⚠️ No lead found for phone number: {phone_number}")
                
                # Create session record with custom config
                new_session = CallSession(
                    session_id=session_id,
                    called_number=phone_number,
                    lead_id=lead.id if lead else None,
                    campaign_id=None,  # Manual call
                    owner_id=current_user.id,  # Set owner to current authenticated user
                    status=CallStatus.DIALING,
                    started_at=datetime.utcnow()
                )
                db.add(new_session)
                db.commit()
            finally:
                db.close()
            
            return {
                "status": "success",
                "session_id": session_id,
                "phone_number": phone_number,
                "message": "Outbound call initiated through Gate VoIP",
                "custom_config": custom_config
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


@app.get("/api/latency")
async def get_latency_stats():
    """
    Get comprehensive latency statistics for all active sessions.
    
    Tracks timing at each stage of the voice pipeline:
    - RTP Receive: Audio packet received from SIM gate (Bulgaria)
    - Decode: After μ-law/A-law decoding  
    - Preprocess: After audio preprocessing pipeline (bandpass, AGC, etc.)
    - VAD: After Voice Activity Detection filtering
    - Gemini Send: When audio is sent to Gemini API
    - Gemini Response: First audio byte received from Gemini
    - Convert: After converting response to telephony format
    - RTP Send: When response is sent back to SIM gate
    
    Network topology:
    - SIM Gate (Bulgaria) <-> UK VPS (Agent) <-> Google Gemini API
    """
    try:
        latency_data = {
            "status": "success",
            "active_sessions": len(active_sessions),
            "sessions": []
        }
        
        for session_id, session_info in active_sessions.items():
            voice_session = session_info["voice_session"]
            
            # Try to get latency tracker from RTP session
            rtp_session = None
            for sid, rtp_sess in rtp_server.sessions.items():
                if hasattr(rtp_sess, 'voice_session') and rtp_sess.voice_session == voice_session:
                    rtp_session = rtp_sess
                    break
            
            session_latency = {
                "session_id": session_id,
                "caller_id": voice_session.caller_id,
                "duration_sec": round((datetime.now(timezone.utc) - session_info["call_start"]).total_seconds(), 2),
                "latency": None
            }
            
            if rtp_session and hasattr(rtp_session, 'latency_tracker'):
                session_latency["latency"] = rtp_session.latency_tracker.get_latency_summary()
            
            latency_data["sessions"].append(session_latency)
        
        # If no active sessions, return summary message
        if not latency_data["sessions"]:
            latency_data["message"] = "No active sessions. Latency data is available during active calls."
        
        return latency_data
        
    except Exception as e:
        logger.error(f"Error getting latency stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/latency/{session_id}")
async def get_session_latency(session_id: str):
    """
    Get detailed latency statistics for a specific session.
    
    Returns comprehensive timing data including:
    - Inbound latency (SIM Gate -> Agent -> Gemini)
    - Outbound latency (Gemini -> Agent -> SIM Gate)
    - Gemini API roundtrip time
    - End-to-end user speech to AI response time
    - Bottleneck analysis
    """
    try:
        # Find the session
        session_info = active_sessions.get(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        voice_session = session_info["voice_session"]
        
        # Find RTP session with latency tracker
        rtp_session = None
        for sid, rtp_sess in rtp_server.sessions.items():
            if hasattr(rtp_sess, 'voice_session') and rtp_sess.voice_session == voice_session:
                rtp_session = rtp_sess
                break
        
        if not rtp_session or not hasattr(rtp_session, 'latency_tracker'):
            raise HTTPException(status_code=404, detail=f"Latency tracker not available for session {session_id}")
        
        summary = rtp_session.latency_tracker.get_latency_summary()
        
        # Add bottleneck analysis
        bottleneck = None
        max_avg = 0
        for stage, stats in summary.get('stages', {}).items():
            if stats.get('avg', 0) > max_avg and 'total' not in stage:
                max_avg = stats['avg']
                bottleneck = stage
        
        summary['bottleneck'] = {
            'stage': bottleneck,
            'avg_latency_ms': max_avg,
            'recommendation': None
        }
        
        if bottleneck:
            if 'gemini' in bottleneck.lower():
                summary['bottleneck']['recommendation'] = "Gemini API latency is expected (200-500ms). Consider closer region or model optimization."
            elif 'preprocess' in bottleneck.lower():
                summary['bottleneck']['recommendation'] = "Audio preprocessing is slow. Consider reducing FFT size or optimizing filters."
            elif 'vad' in bottleneck.lower():
                summary['bottleneck']['recommendation'] = "VAD processing is slow. Consider reducing min_speech_frames."
            elif 'rtp' in bottleneck.lower():
                summary['bottleneck']['recommendation'] = "Network latency to SIM gate. Consider closer server or VPN optimization."
        
        return {
            "status": "success",
            "session_id": session_id,
            "caller_id": voice_session.caller_id,
            "latency": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session latency: {e}")
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
        "transcription_available": TRANSCRIPTION_AVAILABLE,
        "transcription_method": TRANSCRIPTION_METHOD if TRANSCRIPTION_AVAILABLE else None,
        "transcription_model": audio_transcriber.model_size if TRANSCRIPTION_AVAILABLE else None,
        "greeting_generator_available": GREETING_GENERATOR_AVAILABLE,
        "greeting_generator_method": GREETING_GENERATOR_METHOD if GREETING_GENERATOR_AVAILABLE else None,
        "audio_preprocessing": {
            "enabled": True,
            "features": [
                "Advanced Voice Activity Detection (VAD)",
                "Energy-based speech detection",
                "Zero-Crossing Rate analysis",
                "Silence/noise filtering (complete cutoff)",
                "Pre-emphasis filtering",
                "Bandpass filtering (300-3400 Hz)",
                "Spectral noise reduction",
                "Dynamic range compression",
                "Audio normalization"
            ],
            "purpose": "Only speech sent to Gemini - silence and noise completely filtered out",
            "latency_optimizations": [
                "Reduced polling intervals (5ms)",
                "Faster call establishment (1s delay)",
                "Optimized audio buffering (40ms chunks)",
                "Immediate speech transmission"
            ]
        },
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

@app.get("/api/recordings")
async def get_recordings(current_user: User = Depends(get_current_user)):
    """Get list of available call recordings from local sessions (filtered by user ownership)"""
    try:
        # Import necessary modules for session querying
        from crm_database import get_session as get_db_session, CallSession, UserRole, UserManager
        
        # Get all accessible sessions based on user role
        db_session = get_db_session()
        try:
            accessible_sessions = []
            
            if current_user.role == UserRole.SUPERADMIN:
                # Superadmin sees all sessions
                accessible_sessions = db_session.query(CallSession).all()
            elif current_user.role == UserRole.ADMIN:
                # Admin sees their own sessions + their agents' sessions
                user_manager = UserManager(db_session)
                agents = user_manager.get_agents_by_admin(current_user.id)
                agent_ids = [agent.id for agent in agents]
                accessible_user_ids = [current_user.id] + agent_ids
                accessible_sessions = db_session.query(CallSession).filter(CallSession.owner_id.in_(accessible_user_ids)).all()
            else:  # AGENT
                # Agent only sees their own sessions
                accessible_sessions = db_session.query(CallSession).filter(CallSession.owner_id == current_user.id).all()
            
            # Create a set of accessible session IDs for filtering
            accessible_session_ids = {session.session_id for session in accessible_sessions}
        finally:
            db_session.close()
        
        # Get recordings from local sessions directory
        sessions_dir = Path("sessions")
        local_recordings = []
        
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    session_info_path = session_dir / "session_info.json"
                    if session_info_path.exists():
                        try:
                            with open(session_info_path, 'r') as f:
                                session_info = json.load(f)
                            
                            session_id = session_info.get('session_id')
                            
                            # Filter by ownership - only include accessible sessions
                            if session_id not in accessible_session_ids:
                                continue
                            
                            # Check if audio files exist
                            audio_files = {}
                            for audio_type, filename in session_info.get('files', {}).items():
                                audio_path = session_dir / filename
                                if audio_path.exists():
                                    audio_files[audio_type] = {
                                        "filename": filename,
                                        "size_mb": round(audio_path.stat().st_size / (1024 * 1024), 2),
                                        "path": str(audio_path)
                                    }
                            
                            if audio_files:
                                recording_entry = {
                                    "session_id": session_id,
                                    "caller_id": session_info.get('caller_id'),
                                    "called_number": session_info.get('called_number'),
                                    "start_time": session_info.get('start_time'),
                                    "end_time": session_info.get('end_time'),
                                    "duration_seconds": session_info.get('duration_seconds'),
                                    "audio_files": audio_files,
                                    "has_recording": True,
                                    "source": "local"
                                }
                                local_recordings.append(recording_entry)
                        except Exception as e2:
                            logger.warning(f"Error reading local session {session_dir.name}: {e2}")
                            continue
        
        local_recordings.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return {
            "status": "success",
            "total_recordings": len(local_recordings),
            "recordings": local_recordings
        }
        
    except Exception as e:
        logger.error(f"Error getting recordings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch recordings: {str(e)}")

@app.get("/api/transcripts/{session_id}")
async def get_session_transcripts(session_id: str):
    """Get transcripts for a specific session - reads from MongoDB or files"""
    try:
        transcripts = {}
        session_info_data = {}
        
        # First, try to get transcripts from MongoDB
        try:
            from crm_database_mongodb import MongoDB
            db = MongoDB.get_db()
            call_session_doc = db.call_sessions.find_one({"session_id": session_id})
            
            if call_session_doc:
                # Get transcripts from MongoDB
                mongo_transcripts = call_session_doc.get('transcripts', {})
                if mongo_transcripts:
                    transcripts = mongo_transcripts
                    logger.info(f"📖 Loaded {len(transcripts)} transcripts from MongoDB for session {session_id}")
                
                # Get session info from MongoDB
                if call_session_doc.get('session_info'):
                    session_info_data = call_session_doc['session_info']
        except Exception as e:
            logger.warning(f"Could not load transcripts from MongoDB: {e}")
        
        # If no transcripts from MongoDB, try loading from files
        if not transcripts:
            session_dir = Path(f"sessions/{session_id}")
            
            if not session_dir.exists():
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Load session info from file
            session_info_path = session_dir / "session_info.json"
            if session_info_path.exists():
                with open(session_info_path, 'r', encoding='utf-8') as f:
                    session_info_data = json.load(f)
            
            # Get transcript information from session_info if available
            transcripts_info = session_info_data.get('transcripts', {})
            
            # Check for actual transcript files directly (real-time check)
            audio_types = ['incoming', 'outgoing', 'mixed']
            for audio_type in audio_types:
                # Look for .txt files matching the audio type
                txt_files = list(session_dir.glob(f"*{audio_type}*.txt"))
                
                # Filter out non-transcript files (like transcript_*.json)
                txt_files = [f for f in txt_files if f.suffix == '.txt' and 'transcript' not in f.stem or f.stem.startswith(audio_type)]
                
                if txt_files:
                    # Use the first matching file
                    transcript_path = txt_files[0]
                    
                    # Read transcript content
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript_content = f.read()
                    
                    # Get metadata from session_info if available, otherwise use defaults
                    transcript_info = transcripts_info.get(audio_type, {})
                    
                    # Parse metadata from file if it exists in the standard format
                    lines = transcript_content.split('\n')
                    file_metadata = {}
                    if lines and lines[0].startswith('Audio File:'):
                        # Parse the header
                        for line in lines[:10]:  # Check first 10 lines for metadata
                            if ':' in line and not line.startswith('-'):
                                key, value = line.split(':', 1)
                                file_metadata[key.strip()] = value.strip()
                        # Find the separator line
                        separator_idx = next((i for i, line in enumerate(lines) if line.startswith('---')), -1)
                        if separator_idx >= 0 and separator_idx + 1 < len(lines):
                            # Content starts after the separator
                            transcript_content = '\n'.join(lines[separator_idx + 1:]).strip()
                        
                        transcripts[audio_type] = {
                        "filename": transcript_path.name,
                        "language": transcript_info.get('language') or file_metadata.get('Language', 'unknown'),
                        "text_length": len(transcript_content),
                            "confidence": transcript_info.get('confidence'),
                        "transcribed_at": transcript_info.get('transcribed_at') or file_metadata.get('Transcribed'),
                        "success": True,
                        "content": transcript_content,
                        "method": file_metadata.get('Method', transcript_info.get('method', 'unknown'))
                    }
        
        return {
            "status": "success",
            "session_id": session_id,
            "session_info": {
                "caller_id": session_info_data.get('caller_id'),
                "called_number": session_info_data.get('called_number'),
                "start_time": session_info_data.get('start_time'),
                "end_time": session_info_data.get('end_time'),
                "duration_seconds": session_info_data.get('duration_seconds')
            },
            "transcripts": transcripts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcripts for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcripts/{session_id}/retranscribe")
async def retranscribe_session(session_id: str, background_tasks: BackgroundTasks):
    """Trigger re-transcription for a specific session"""
    try:
        if not TRANSCRIPTION_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Transcription service unavailable - no transcription method available"
            )
        
        session_dir = Path(f"sessions/{session_id}")
        
        if not session_dir.exists():
            # Session directory doesn't exist, but check if we have asterisk_linkedid in database
            try:
                from crm_database_mongodb import CallSession, MongoDB
                db = MongoDB.get_db()
                call_session_doc = db.call_sessions.find_one({"session_id": session_id})
                
                if call_session_doc and call_session_doc.get('asterisk_linkedid'):
                    # We have a linkedid, create session directory and proceed with PBX recording
                    session_dir.mkdir(parents=True, exist_ok=True)
                    asterisk_linkedid = call_session_doc['asterisk_linkedid']
                    
                    # Add background task for PBX transcription
                    background_tasks.add_task(
                        _background_transcribe_pbx_recording,
                        session_dir,
                        session_id,
                        asterisk_linkedid
                    )
                    
                    return {
                        "status": "success",
                        "message": f"Transcription started for PBX recording (linked_id: {asterisk_linkedid})",
                        "session_id": session_id,
                        "source": "pbx"
                    }
                else:
                    raise HTTPException(status_code=404, detail="Session not found and no PBX recording available")
            except Exception as e:
                logger.error(f"Error checking for PBX recording: {e}")
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if we should use PBX recording (asterisk_linkedid available)
        try:
            from crm_database_mongodb import CallSession, MongoDB
            db = MongoDB.get_db()
            call_session_doc = db.call_sessions.find_one({"session_id": session_id})
            
            if call_session_doc and call_session_doc.get('asterisk_linkedid'):
                asterisk_linkedid = call_session_doc['asterisk_linkedid']
                logger.info(f"🔗 Found asterisk_linkedid: {asterisk_linkedid}, using PBX recording")
                
                # Add background task for PBX transcription
                background_tasks.add_task(
                    _background_transcribe_pbx_recording,
                    session_dir,
                    session_id,
                    asterisk_linkedid
                )
                
                return {
                    "status": "success",
                    "message": f"Transcription started for PBX recording (linked_id: {asterisk_linkedid})",
                    "session_id": session_id,
                    "source": "pbx"
                }
        except Exception as e:
            logger.warning(f"Could not check for asterisk_linkedid: {e}, falling back to local files")
        
        # Fallback to local audio files
        # Check if audio files exist
        audio_files_found = []
        for pattern in ["*incoming*.wav", "*outgoing*.wav", "*mixed*.wav"]:
            audio_files_found.extend(list(session_dir.glob(pattern)))
        
        if not audio_files_found:
            raise HTTPException(status_code=404, detail="No audio files found for transcription")
        
        # Load session info to get caller ID for language detection
        session_info_path = session_dir / "session_info.json"
        caller_id = "Unknown"
        if session_info_path.exists():
            with open(session_info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
                caller_id = session_info.get('caller_id', 'Unknown')
        
        # Add background task for transcription
        background_tasks.add_task(
            _background_transcribe_session,
            session_dir,
            caller_id
        )
        
        return {
            "status": "success",
            "message": f"Transcription started for session {session_id}",
            "session_id": session_id,
            "audio_files_found": len(audio_files_found),
            "source": "local"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting retranscription for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcripts/{session_id}/generate_summary")
async def generate_summary(session_id: str, request: GenerateSummaryRequest, background_tasks: BackgroundTasks):
    """Generate AI summary for a specific session using transcripts from MongoDB or files"""
    try:
        # First, try to get transcripts from MongoDB
        transcripts_content = []
        try:
            from crm_database_mongodb import MongoDB
            db = MongoDB.get_db()
            call_session_doc = db.call_sessions.find_one({"session_id": session_id})
            
            if call_session_doc and call_session_doc.get('transcripts'):
                mongo_transcripts = call_session_doc['transcripts']
                for audio_type, transcript_data in mongo_transcripts.items():
                    if transcript_data.get('content'):
                        transcripts_content.append(transcript_data['content'])
                
                if transcripts_content:
                    logger.info(f"📖 Using transcripts from MongoDB for summary generation")
                    # Add background task with MongoDB transcripts
                    background_tasks.add_task(
                        _background_generate_summary_from_content,
                        session_id,
                        transcripts_content,
                        request.language
                    )
                    
                    return {
                        "status": "success",
                        "message": f"Summary generation started for session {session_id}",
                        "session_id": session_id,
                        "transcripts_found": len(transcripts_content),
                        "language": request.language,
                        "source": "mongodb"
                    }
        except Exception as e:
            logger.warning(f"Could not load transcripts from MongoDB: {e}")
        
        # Fallback to file-based transcripts
        session_dir = Path(f"sessions/{session_id}")
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if transcript files exist (both new format *_transcript.txt and old format with timestamps)
        transcript_files = []
        for pattern in ["*_transcript.txt", "incoming_*.txt", "outgoing_*.txt", "mixed_*.txt"]:
            files = list(session_dir.glob(pattern))
            # Filter to only include .txt files (exclude .wav files)
            transcript_files.extend([f for f in files if f.suffix == '.txt'])
        
        # Remove duplicates by converting to set and back
        transcript_files = list(set(transcript_files))
        
        if not transcript_files:
            raise HTTPException(status_code=404, detail="No transcript files found. Please transcribe the session first.")
        
        # Add background task for summary generation with language parameter
        background_tasks.add_task(
            _background_generate_summary,
            session_dir,
            transcript_files,
            request.language
        )
        
        return {
            "status": "success",
            "message": f"Summary generation started for session {session_id}",
            "session_id": session_id,
            "transcript_files_found": len(transcript_files),
            "language": request.language,
            "source": "files"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting summary generation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcripts/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get AI summary for a specific session - reads from MongoDB or files"""
    try:
        # First, try to get summary from MongoDB
        try:
            from crm_database_mongodb import MongoDB
            db = MongoDB.get_db()
            call_session_doc = db.call_sessions.find_one({"session_id": session_id})
            
            if call_session_doc and call_session_doc.get('analysis'):
                summary_data = call_session_doc['analysis']
                logger.info(f"📊 Loaded summary from MongoDB for session {session_id}")
                return {
                    "status": "success",
                    "session_id": session_id,
                    "summary": summary_data
                }
        except Exception as e:
            logger.warning(f"Could not load summary from MongoDB: {e}")
        
        # If not in MongoDB, try loading from file
        session_dir = Path(f"sessions/{session_id}")
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Look for summary file
        summary_path = session_dir / "analysis_result.json"
        
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Summary not found. Please generate a summary first.")
        
        # Read summary content
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        return {
            "status": "success",
            "session_id": session_id,
            "summary": summary_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcripts/{session_id}/{audio_type}")
async def get_specific_transcript(session_id: str, audio_type: str):
    """Get transcript content for specific audio type (incoming, outgoing, mixed)"""
    try:
        if audio_type not in ['incoming', 'outgoing', 'mixed']:
            raise HTTPException(status_code=400, detail="Invalid audio type. Must be 'incoming', 'outgoing', or 'mixed'")
        
        session_dir = Path(f"sessions/{session_id}")
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Look for transcript file
        transcript_path = session_dir / f"{audio_type}_transcript.txt"
        
        if not transcript_path.exists():
            raise HTTPException(status_code=404, detail=f"Transcript not found for {audio_type} audio")
        
        # Read transcript content
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get transcript metadata from session info
        session_info_path = session_dir / "session_info.json"
        transcript_info = {}
        if session_info_path.exists():
            with open(session_info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
                transcripts_info = session_info.get('transcripts', {})
                transcript_info = transcripts_info.get(audio_type, {})
        
        return {
            "status": "success",
            "session_id": session_id,
            "audio_type": audio_type,
            "transcript": {
                "content": content,
                "filename": f"{audio_type}_transcript.txt",
                "language": transcript_info.get('language', 'unknown'),
                "text_length": transcript_info.get('text_length', len(content)),
                "confidence": transcript_info.get('confidence'),
                "transcribed_at": transcript_info.get('transcribed_at'),
                "success": transcript_info.get('success', True)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcript for session {session_id}, audio type {audio_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sms/send")
async def send_sms(request: SendSMSRequest):
    """
    Send SMS via the PBX system using the SIM card gateway.
    
    Parameters:
    - phone_number: Target phone number (e.g., "0888123456")
    - message: SMS content to send
    - gate_slot: Gate slot number (9-19) to use as sender
    """
    # Suppress SSL warnings for internal PBX
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        # Validate gate_slot
        if request.gate_slot < 9 or request.gate_slot > 19:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid gate slot {request.gate_slot}. Must be between 9 and 19"
            )
        
        # Clean phone number (remove spaces, dashes)
        phone = request.phone_number.strip().replace(" ", "").replace("-", "")
        
        # Validate phone number format (basic validation)
        if not phone or len(phone) < 8:
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format"
            )
        
        # URL encode the message
        import urllib.parse
        encoded_message = urllib.parse.quote(request.message)
        
        # Build the SMS API URL
        sms_api_url = (
            f"https://pbx.voipsystems.bg/public/sms_evolvs.php"
            f"?api_key=GytVjPdThdfsdo29bngbut8so-AI-rt63go06ty22"
            f"&from={request.gate_slot}"
            f"&phone={phone}"
            f"&sms={encoded_message}"
        )
        
        logger.info(f"📱 Sending SMS to {phone} via gate slot {request.gate_slot}")
        logger.info(f"📝 Message length: {len(request.message)} characters")
        logger.info(f"🔗 SMS API URL: {sms_api_url}")
        
        # Make the request to the SMS gateway
        # verify=False for internal PBX systems that may have self-signed certs
        response = requests.get(sms_api_url, timeout=(10, 60), verify=False)
        
        logger.info(f"📡 SMS gateway response status: {response.status_code}")
        logger.info(f"📡 SMS gateway response: {response.text[:500] if response.text else 'empty'}")
        
        if response.status_code == 200:
            logger.info(f"✅ SMS sent successfully to {phone}")
            return {
                "status": "success",
                "message": "SMS sent successfully",
                "phone_number": phone,
                "gate_slot": request.gate_slot,
                "sms_length": len(request.message),
                "response": response.text
            }
        else:
            logger.error(f"❌ SMS gateway returned status {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"SMS gateway error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        logger.error("❌ SMS gateway request timed out")
        raise HTTPException(status_code=504, detail="SMS gateway request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ SMS gateway connection error: {e}")
        raise HTTPException(status_code=502, detail=f"SMS gateway connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error sending SMS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _background_transcribe_session(session_dir: Path, caller_id: str):
    """Background task function for transcribing a session using the fast Gemini script"""
    try:
        logger.info(f"🎤 Background transcription started for {session_dir.name}")
        logger.info(f"🚀 Using fast Gemini API transcription script")
        logger.info(f"🌍 Using automatic language detection (no language hint)")
        
        # Call the transcribe_audio.py script using subprocess
        import subprocess
        import sys
        
        # No language hint - let Gemini auto-detect the language(s)
        cmd = [sys.executable, "transcribe_audio.py", str(session_dir), "--quiet"]
        
        logger.info(f"Running transcription command: {' '.join(cmd)}")
        
        # Run the script
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Force UTF-8 encoding (Windows default is cp1252)
            errors='replace',   # Replace invalid characters instead of crashing
            timeout=300  # 5 minute timeout (should be more than enough with Gemini)
        )
        
        if process.returncode == 0:
            logger.info(f"✅ Transcription script completed successfully")
            logger.info(f"✅ Background transcription completed for {session_dir.name}")
            
            # Save transcripts to MongoDB
            try:
                from session_mongodb_helper import save_transcripts_to_mongodb
                session_id = session_dir.name
                save_transcripts_to_mongodb(session_id, session_dir)
            except Exception as mongo_error:
                logger.warning(f"⚠️  Could not save transcripts to MongoDB: {mongo_error}")
        else:
            logger.error(f"❌ Transcription script failed with return code {process.returncode}")
            logger.error(f"stderr: {process.stderr}")
            raise Exception(f"Transcription script failed: {process.stderr}")
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Transcription timed out for {session_dir.name}")
    except Exception as e:
        logger.error(f"❌ Error in background transcription for {session_dir.name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def _background_transcribe_pbx_recording(session_dir: Path, session_id: str, asterisk_linkedid: str):
    """Background task function for transcribing PBX recording (single MP3 file)"""
    try:
        logger.info(f"🎤 Background PBX transcription started for {session_id}")
        logger.info(f"🔗 Asterisk Linked ID: {asterisk_linkedid}")
        
        # Construct PBX recording URL
        pbx_url = f"http://192.168.50.50/play.php?api=R0SHJIU9w55wRR&uniq={asterisk_linkedid}"
        logger.info(f"📥 Downloading recording from: {pbx_url}")
        
        # Download the MP3 file temporarily
        import requests
        import tempfile
        
        temp_mp3_file = None
        try:
            # Download the recording
            response = requests.get(pbx_url, timeout=60)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp3', delete=False) as temp_file:
                temp_mp3_file = temp_file.name
                temp_file.write(response.content)
            
            logger.info(f"✅ Downloaded recording to temporary file: {temp_mp3_file}")
            
            # Transcribe using Gemini API
            from transcribe_audio import AudioTranscriber
            transcriber = AudioTranscriber()
            
            if not transcriber.available:
                raise Exception("Transcription service not available")
            
            logger.info(f"🚀 Starting transcription of PBX recording...")
            
            # Transcribe with updated prompt for conversation
            result = transcriber.transcribe_audio_file(
                temp_mp3_file, 
                language=None,  # Auto-detect language
                is_conversation=True  # Flag to use conversation-specific prompt
            )
            
            if result.get('success'):
                transcript_text = result.get('text', '')
                conversation_data = result.get('conversation')
                logger.info(f"✅ Transcription completed successfully")
                logger.info(f"📝 Transcript length: {len(transcript_text)} characters")
                if conversation_data:
                    logger.info(f"💬 Conversation has {len(conversation_data)} turns")
                
                # Save transcript to session directory as "mixed" (since it's a full conversation)
                transcript_file = session_dir / "mixed_transcript.txt"
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                
                logger.info(f"💾 Saved transcript to: {transcript_file}")
                
                # Update session_info.json
                session_info_path = session_dir / "session_info.json"
                session_info = {}
                if session_info_path.exists():
                    with open(session_info_path, 'r', encoding='utf-8') as f:
                        session_info = json.load(f)
                
                session_info['transcription_completed'] = True
                session_info['transcription_source'] = 'pbx'
                session_info['asterisk_linkedid'] = asterisk_linkedid
                session_info['transcribed_at'] = datetime.now().isoformat()
                session_info['has_conversation_structure'] = conversation_data is not None
                
                with open(session_info_path, 'w', encoding='utf-8') as f:
                    json.dump(session_info, f, indent=2, ensure_ascii=False)
                
                # Save transcripts to MongoDB with conversation structure
                try:
                    from session_mongodb_helper import save_transcripts_to_mongodb_with_conversation
                    save_transcripts_to_mongodb_with_conversation(
                        session_id, 
                        session_dir,
                        conversation_data
                    )
                    logger.info(f"✅ Saved transcript to MongoDB")
                except Exception as mongo_error:
                    logger.warning(f"⚠️ Could not save transcript to MongoDB: {mongo_error}")
                    # Fallback to regular save
                    try:
                        from session_mongodb_helper import save_transcripts_to_mongodb
                        save_transcripts_to_mongodb(session_id, session_dir)
                    except:
                        pass
                
            else:
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"❌ Transcription failed: {error_msg}")
                raise Exception(f"Transcription failed: {error_msg}")
                
        finally:
            # Clean up temporary MP3 file
            if temp_mp3_file and os.path.exists(temp_mp3_file):
                try:
                    os.unlink(temp_mp3_file)
                    logger.info(f"🗑️ Deleted temporary file: {temp_mp3_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete temporary file: {e}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error downloading PBX recording: {e}")
    except Exception as e:
        logger.error(f"❌ Error in PBX transcription for {session_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def _background_generate_summary(session_dir: Path, transcript_files: list, language: str = "English"):
    """Background task function for generating AI summary using the file_summarizer.py script"""
    try:
        logger.info(f"📊 Background summary generation started for {session_dir.name}")
        logger.info(f"📝 Found {len(transcript_files)} transcript files")
        logger.info(f"🌐 Summary language: {language}")
        
        # Call the file_summarizer.py script using subprocess
        import subprocess
        import sys
        import shutil
        
        # Build command with transcript file paths and language parameter
        transcript_paths = [str(f) for f in transcript_files]
        cmd = [sys.executable, "summary/file_summarizer.py", "--language", language] + transcript_paths
        
        logger.info(f"Running summary command: {' '.join(cmd)}")
        
        # Run the script
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120  # 2 minute timeout
        )
        
        if process.returncode == 0:
            logger.info(f"✅ Summary script completed successfully")
            logger.info(f"Script output: {process.stdout}")
            
            # Move the generated analysis_result.json to the session directory
            # The script saves it in the current working directory (root)
            source_file = Path("analysis_result.json")
            dest_file = session_dir / "analysis_result.json"
            
            if source_file.exists():
                # If destination exists, remove it first to ensure replacement
                if dest_file.exists():
                    logger.info(f"🗑️ Removing old summary at {dest_file}")
                    dest_file.unlink()
                
                # Move (not copy) the file to the session directory
                shutil.move(str(source_file), str(dest_file))
                logger.info(f"✅ Summary moved to {dest_file}")
                
                # Save analysis to MongoDB
                try:
                    from session_mongodb_helper import save_analysis_to_mongodb
                    session_id = session_dir.name
                    save_analysis_to_mongodb(session_id, session_dir)
                except Exception as mongo_error:
                    logger.warning(f"⚠️  Could not save analysis to MongoDB: {mongo_error}")
            else:
                logger.error(f"❌ Summary file not found at {source_file}")
                logger.error(f"Working directory: {Path.cwd()}")
                logger.error(f"Files in root: {list(Path('.').glob('*.json'))}")
            
            logger.info(f"✅ Background summary generation completed for {session_dir.name}")
        else:
            logger.error(f"❌ Summary script failed with return code {process.returncode}")
            logger.error(f"stdout: {process.stdout}")
            logger.error(f"stderr: {process.stderr}")
            raise Exception(f"Summary script failed: {process.stderr}")
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Summary generation timed out for {session_dir.name}")
    except Exception as e:
        logger.error(f"❌ Error in background summary generation for {session_dir.name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def _background_generate_summary_from_content(session_id: str, transcripts_content: list, language: str = "English"):
    """Background task function for generating AI summary directly from MongoDB transcript content"""
    try:
        logger.info(f"📊 Background summary generation started for {session_id} (from MongoDB)")
        logger.info(f"📝 Found {len(transcripts_content)} transcripts")
        logger.info(f"🌐 Summary language: {language}")
        
        # Import the summarizer directly
        from summary.file_summarizer import summarize_from_content
        
        # Combine all transcript content
        combined_content = "\n\n".join(transcripts_content)
        
        logger.info(f"📄 Combined transcript length: {len(combined_content)} characters")
        
        # Generate summary
        result = summarize_from_content(combined_content, language=language)
        
        if result:
            logger.info(f"✅ Summary generated successfully")
            logger.info(f"Summary: {result.get('summary', 'N/A')[:100]}...")
            
            # Save analysis to MongoDB
            try:
                from crm_database_mongodb import MongoDB
                db = MongoDB.get_db()
                
                update_data = {
                    'analysis': result,
                    'updated_at': datetime.utcnow()
                }
                
                db.call_sessions.update_one(
                    {"session_id": session_id},
                    {"$set": update_data}
                )
                
                logger.info(f"✅ Saved summary to MongoDB for session {session_id}")
            except Exception as mongo_error:
                logger.warning(f"⚠️ Could not save summary to MongoDB: {mongo_error}")
            
            # Also save to file system for compatibility
            session_dir = Path(f"sessions/{session_id}")
            if session_dir.exists():
                summary_file = session_dir / "analysis_result.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Saved summary to file: {summary_file}")
            
            logger.info(f"✅ Background summary generation completed for {session_id}")
        else:
            logger.error(f"❌ Summary generation returned no result")
        
    except Exception as e:
        logger.error(f"❌ Error in background summary generation for {session_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

@app.get("/api/transcription/status")
async def get_transcription_status():
    """Get transcription service status and capabilities"""
    return {
        "status": "success",
        "transcription_available": TRANSCRIPTION_AVAILABLE,
        "transcription_method": TRANSCRIPTION_METHOD if TRANSCRIPTION_AVAILABLE else None,
        "service_status": "enabled" if TRANSCRIPTION_AVAILABLE else "disabled",
        "model_info": {
            "model_size": audio_transcriber.model_size if TRANSCRIPTION_AVAILABLE else None,
            "model_loaded": audio_transcriber.model_loaded if TRANSCRIPTION_AVAILABLE else False
        } if TRANSCRIPTION_AVAILABLE else None,
        "supported_languages": [
            "en", "bg", "ro", "el", "de", "fr", "es", "it", "ru", "zh", "ja", "ko", "ar", "hi", "pt"
        ] if TRANSCRIPTION_AVAILABLE else [],
        "reason": f"Transcription enabled using {TRANSCRIPTION_METHOD}" if TRANSCRIPTION_AVAILABLE else "No transcription method available - install faster-whisper or configure OpenAI API key"
    }

@app.get("/api/audio/preprocessing")
async def get_audio_preprocessing_status():
    """Get audio preprocessing status and configuration - Professional Cold Calling Pipeline"""
    return {
        "status": "success",
        "preprocessing_enabled": True,
        "pipeline_type": "Professional Cold Calling (ITU-T G.712 Compliant)",
        "sample_rate": 8000,
        "estimated_latency_ms": "7-10",
        "processing_pipeline": [
            {
                "stage": 1,
                "name": "Bandpass Filter",
                "description": "ITU-T G.712 telephony band filter",
                "frequency_range": "300-3400 Hz",
                "filter_type": "4th order Butterworth IIR",
                "removes": "50/60Hz hum, high-frequency noise",
                "latency_ms": "~1"
            },
            {
                "stage": 2,
                "name": "Spectral Noise Suppression",
                "description": "FFT-based noise reduction with adaptive noise floor",
                "method": "Overlap-add spectral subtraction",
                "fft_size": 256,
                "hop_size": 80,
                "reduction_strength": "70%",
                "noise_floor_estimation": "First 80ms of each call",
                "latency_ms": "~5-8"
            },
            {
                "stage": 3,
                "name": "Noise Gate",
                "description": "Soft gate to silence very quiet passages",
                "threshold_db": -42,
                "attack_ms": 5,
                "release_ms": 100,
                "prevents": "Ghost words from background noise"
            },
            {
                "stage": 4,
                "name": "AGC (Automatic Gain Control)",
                "description": "Normalizes voice levels for consistent Gemini input",
                "target_level_db": -16,
                "max_gain_db": 15,
                "min_gain_db": -6,
                "attack_ms": 5,
                "release_ms": 100,
                "handles": "Quiet and loud callers equally well"
            },
            {
                "stage": 5,
                "name": "Soft Limiter",
                "description": "Gentle knee compression to prevent clipping",
                "threshold": 0.9,
                "knee_db": 6,
                "prevents": "Digital distortion on peaks"
            },
            {
                "stage": 6,
                "name": "WebRTC VAD",
                "description": "Voice Activity Detection for speech filtering",
                "aggressiveness": 3,
                "min_speech_frames": 3,
                "min_silence_frames": 15,
                "filters": "Non-speech audio before sending to Gemini"
            },
            {
                "stage": 7,
                "name": "Upsampling",
                "description": "High-quality resampling for Gemini",
                "method": "resample_poly with Kaiser window",
                "from_rate": 8000,
                "to_rate": 24000,
                "window": "Kaiser (beta=5.0)"
            }
        ],
        "telephony_standards": {
            "bandpass": "ITU-T G.712 (300-3400 Hz)",
            "codec_support": ["PCMU (μ-law)", "PCMA (A-law)"],
            "sample_rate": "8000 Hz (telephony standard)"
        },
        "benefits": [
            "🎯 Professional cold calling audio quality",
            "🔇 Background noise/hiss removed via spectral suppression",
            "📈 Consistent voice levels via AGC (handles quiet/loud callers)",
            "⚡ Low latency (~7-10ms) suitable for real-time conversation",
            "🎤 Improved Gemini speech recognition accuracy",
            "✅ Natural conversation flow maintained",
            "🚀 Noise floor adapts to each call's acoustic environment"
        ],
        "note": "Professional 4-stage preprocessing pipeline applied to incoming telephony audio before sending to Gemini. Noise floor is re-estimated at the start of each call for optimal adaptation."
    }

def start_ami_thread():
    """Start Asterisk AMI monitoring in background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Use credentials from Asterisk config
    ami_host = config.get('host', '192.168.50.50')
    ami_port = 5038
    ami_username = 'voipsystems'
    ami_password = 'asccyber@1'
    
    logger.info(f"📡 Connecting to Asterisk AMI at {ami_host}:{ami_port}")
    loop.run_until_complete(start_ami_monitoring(ami_host, ami_port, ami_username, ami_password))

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
    if TRANSCRIPTION_AVAILABLE:
        logger.info(f"🎤 Transcription: {TRANSCRIPTION_METHOD} with '{audio_transcriber.model_size}' model (manual only)")
        logger.info("🇧🇬 Optimized for Bulgarian and multilingual transcription")
        logger.info("   📋 Automatic transcription disabled - available through CRM interface")
    else:
        logger.info("⚠️ Transcription: DISABLED (no method available)")
    logger.info("🎵 Audio Processing: PROFESSIONAL COLD CALLING PIPELINE")
    logger.info("   📊 Stage 1: Bandpass Filter (300-3400Hz ITU-T G.712 telephony standard)")
    logger.info("   🔇 Stage 2: Spectral Noise Suppression (FFT-based, 70% reduction)")
    logger.info("   📈 Stage 3: AGC (Target -16dB, Attack 5ms, Release 100ms)")
    logger.info("   🎚️ Stage 4: Soft Limiter (gentle knee compression)")
    logger.info("   ⚡ Upsampling: resample_poly 8kHz→24kHz with Kaiser window")
    logger.info("   ⏱️ Total additional latency: ~7-10ms for professional quality")
    logger.info("🔊 Greeting System: ENABLED for both incoming and outbound calls")
    logger.info("   🎙️ Plays greeting.wav file automatically when call is answered")
    logger.info("   🤖 AI responds naturally when user speaks (no artificial triggers)")
    logger.info("   ⚡ FULL-DUPLEX MODE: Users can interrupt model at any time (even mid-word!)")
    logger.info("   ⚡ 10ms audio chunks for instant interruption detection (<10ms latency)")
    logger.info("   ⚡ Server-side interruption detection via Gemini Live API")
    logger.info("=" * 60)
    logger.info("Endpoints:")
    logger.info("  GET /health - System health check (includes transcription status)")
    logger.info("  GET /api/config - Current configuration")
    logger.info("  GET /api/sessions - Active call sessions")
    logger.info("  GET /api/recordings - List call recordings (with transcripts)")
    logger.info("  POST /api/make_call - Initiate outbound call")
    logger.info("  GET /api/transcription/status - Check transcription service status")
    logger.info("  GET /api/audio/preprocessing - Check audio preprocessing pipeline status")
    logger.info("  GET /api/transcripts/{session_id} - Get all transcripts for session")
    logger.info("  GET /api/transcripts/{session_id}/{audio_type} - Get specific transcript")
    logger.info("  POST /api/transcripts/{session_id}/retranscribe - Re-transcribe session")
    logger.info("=" * 60)
    
    # Start AMI monitoring thread
    logger.info("📡 Starting Asterisk AMI monitor for linkedid capture...")
    ami_thread = threading.Thread(target=start_ami_thread, daemon=True)
    ami_thread.start()
    logger.info("✅ AMI monitor thread started")
    
    uvicorn.run(
        "windows_voice_agent:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
