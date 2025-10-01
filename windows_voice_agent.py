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
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import queue
from scipy import signal

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pyaudio
import requests
from scipy.signal import resample
import numpy as np

# Multi-tier transcription system with Windows-compatible fallbacks
TRANSCRIPTION_METHOD = None
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False
OPENAI_API_AVAILABLE = False

# First try faster-whisper (Windows-friendly, no numba dependency)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    TRANSCRIPTION_METHOD = "faster_whisper"
    logger.info("✅ faster-whisper library loaded successfully (recommended for Windows)")
except ImportError:
    logger.info("💡 faster-whisper not available, trying openai-whisper...")
except Exception as e:
    logger.warning(f"⚠️ faster-whisper failed to load: {e}")

# Try original openai-whisper first (local, better for Bulgarian)
if not FASTER_WHISPER_AVAILABLE:
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

# Fallback to OpenAI API only if local whisper is not available
if not FASTER_WHISPER_AVAILABLE and not WHISPER_AVAILABLE:
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
TRANSCRIPTION_AVAILABLE = FASTER_WHISPER_AVAILABLE or OPENAI_API_AVAILABLE or WHISPER_AVAILABLE
if TRANSCRIPTION_AVAILABLE:
    logger.info(f"🎤 Transcription enabled using: {TRANSCRIPTION_METHOD}")
else:
    logger.warning("⚠️ No transcription method available - transcription features will be disabled")
    logger.warning("💡 To enable transcription on Windows, try: pip install faster-whisper")
    logger.warning("💡 Or configure OpenAI API: set OPENAI_API_KEY environment variable")

class AudioPreprocessor:
    """Advanced audio preprocessing for noise reduction and speech enhancement"""
    
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * 0.025)  # 25ms frames
        self.hop_size = int(sample_rate * 0.010)   # 10ms hop
        
        # Noise estimation parameters
        self.noise_estimate = None
        self.alpha = 0.95  # Smoothing factor for noise estimation
        self.noise_floor = 0.01  # Minimum noise floor
        
        # Speech enhancement parameters
        self.pre_emphasis = 0.97
        self.previous_sample = 0
        
        # Voice activity detection
        self.energy_threshold = None
        self.energy_history = []
        self.max_history = 30  # 300ms of history at 10ms frames
        
        # Dynamic range compression
        self.compressor_threshold = 0.7
        self.compressor_ratio = 4.0
        self.compressor_attack = 0.003  # 3ms
        self.compressor_release = 0.1   # 100ms
        self.compressor_envelope = 0.0
        
        logger.info("🎵 AudioPreprocessor initialized for speech enhancement")
    
    def process_audio(self, audio_data: bytes) -> bytes:
        """
        Apply comprehensive audio preprocessing to improve transcription accuracy.
        
        Args:
            audio_data: Raw PCM audio data (16-bit, mono, 8kHz)
            
        Returns:
            Processed audio data with noise reduction and enhancement
        """
        try:
            if len(audio_data) < 2:
                return audio_data
                
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_array) == 0:
                return audio_data
                
            # Step 1: Pre-emphasis filter (enhance high frequencies)
            audio_array = self._apply_pre_emphasis(audio_array)
            
            # Step 2: Bandpass filter (focus on speech frequencies)
            audio_array = self._apply_bandpass_filter(audio_array)
            
            # Step 3: Voice activity detection and noise gate
            audio_array = self._apply_noise_gate(audio_array)
            
            # Step 4: Spectral noise reduction
            audio_array = self._spectral_noise_reduction(audio_array)
            
            # Step 5: Dynamic range compression
            audio_array = self._apply_compression(audio_array)
            
            # Step 6: Normalize audio levels
            audio_array = self._normalize_audio(audio_array)
            
            # Convert back to int16
            audio_array = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
            
            return audio_array.tobytes()
            
        except Exception as e:
            logger.warning(f"⚠️ Audio preprocessing failed, using original: {e}")
            return audio_data
    
    def _apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies"""
        try:
            if len(audio) == 0:
                return audio
                
            # Simple high-pass filter: y[n] = x[n] - α * x[n-1]
            emphasized = np.zeros_like(audio)
            emphasized[0] = audio[0]
            
            for i in range(1, len(audio)):
                emphasized[i] = audio[i] - self.pre_emphasis * audio[i-1]
            
            return emphasized
            
        except Exception as e:
            logger.debug(f"Pre-emphasis failed: {e}")
            return audio
    
    def _apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter for telephony speech frequencies (300-3400 Hz)"""
        try:
            if len(audio) < 10:  # Need minimum samples for filtering
                return audio
                
            # Design bandpass filter for speech frequencies
            nyquist = self.sample_rate / 2
            low_freq = 300 / nyquist   # Normalize frequencies
            high_freq = 3400 / nyquist
            
            # Ensure frequencies are valid
            low_freq = max(0.01, min(low_freq, 0.99))
            high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
            
            # Create bandpass filter
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            
            # Apply zero-phase filtering to avoid delay
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Bandpass filtering failed: {e}")
            return audio
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply voice activity detection and noise gating"""
        try:
            if len(audio) == 0:
                return audio
                
            # Calculate energy
            energy = np.mean(audio ** 2)
            
            # Update energy history
            self.energy_history.append(energy)
            if len(self.energy_history) > self.max_history:
                self.energy_history.pop(0)
            
            # Set threshold based on recent history
            if self.energy_threshold is None and len(self.energy_history) >= 10:
                # Initial threshold: median energy * 2
                self.energy_threshold = np.median(self.energy_history) * 2.0
            elif len(self.energy_history) >= self.max_history:
                # Adaptive threshold
                recent_min = np.percentile(self.energy_history, 20)  # 20th percentile
                self.energy_threshold = max(recent_min * 3.0, self.noise_floor)
            
            # Apply noise gate
            if self.energy_threshold is not None and energy < self.energy_threshold:
                # Reduce noise segments by 80%
                audio = audio * 0.2
            
            return audio
            
        except Exception as e:
            logger.debug(f"Noise gate failed: {e}")
            return audio
    
    def _spectral_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction for noise reduction"""
        try:
            if len(audio) < self.frame_size:
                return audio
                
            # Frame the signal
            frames = []
            for i in range(0, len(audio) - self.frame_size + 1, self.hop_size):
                frame = audio[i:i + self.frame_size]
                if len(frame) == self.frame_size:
                    frames.append(frame)
            
            if not frames:
                return audio
                
            enhanced_frames = []
            
            for frame in frames:
                # Apply window
                windowed = frame * np.hanning(len(frame))
                
                # FFT
                fft = np.fft.rfft(windowed)
                magnitude = np.abs(fft)
                phase = np.angle(fft)
                
                # Update noise estimate (use first few frames for noise estimation)
                if self.noise_estimate is None:
                    self.noise_estimate = magnitude
                else:
                    # Adaptive noise estimation
                    is_speech = np.mean(magnitude) > np.mean(self.noise_estimate) * 2
                    if not is_speech:
                        # Update noise estimate during quiet periods
                        self.noise_estimate = self.alpha * self.noise_estimate + (1 - self.alpha) * magnitude
                
                # Spectral subtraction
                enhanced_magnitude = magnitude - 0.5 * self.noise_estimate
                enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
                
                # Reconstruct signal
                enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
                enhanced_frame = np.fft.irfft(enhanced_fft)
                
                # Remove windowing effect
                if len(enhanced_frame) == self.frame_size:
                    enhanced_frames.append(enhanced_frame)
            
            if not enhanced_frames:
                return audio
                
            # Overlap-add reconstruction
            output_length = len(frames) * self.hop_size + self.frame_size - self.hop_size
            enhanced_audio = np.zeros(output_length)
            
            for i, frame in enumerate(enhanced_frames):
                start = i * self.hop_size
                end = start + len(frame)
                if end <= len(enhanced_audio):
                    enhanced_audio[start:end] += frame
            
            # Trim to original length
            enhanced_audio = enhanced_audio[:len(audio)]
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Spectral noise reduction failed: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression to even out volume levels"""
        try:
            if len(audio) == 0:
                return audio
                
            compressed = np.zeros_like(audio)
            
            for i, sample in enumerate(audio):
                # Calculate envelope
                input_level = abs(sample)
                
                if input_level > self.compressor_envelope:
                    # Attack
                    self.compressor_envelope += (input_level - self.compressor_envelope) * self.compressor_attack
                else:
                    # Release
                    self.compressor_envelope += (input_level - self.compressor_envelope) * self.compressor_release
                
                # Calculate gain reduction
                if self.compressor_envelope > self.compressor_threshold:
                    excess = self.compressor_envelope - self.compressor_threshold
                    gain_reduction = 1.0 - (excess * (1.0 - 1.0/self.compressor_ratio))
                    compressed[i] = sample * gain_reduction
                else:
                    compressed[i] = sample
            
            return compressed
            
        except Exception as e:
            logger.debug(f"Compression failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels while preserving dynamics"""
        try:
            if len(audio) == 0:
                return audio
                
            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms > 1e-6:  # Avoid division by zero
                # Target RMS level (around -20 dBFS)
                target_rms = 0.1
                gain = target_rms / rms
                
                # Limit gain to prevent excessive amplification
                gain = min(gain, 10.0)  # Max 20dB gain
                
                normalized = audio * gain
                
                # Soft limiting to prevent clipping
                normalized = np.tanh(normalized * 0.9) / 0.9
                
                return normalized
            
            return audio
            
        except Exception as e:
            logger.debug(f"Normalization failed: {e}")
            return audio

# Import our existing voice agent components
from main import SessionLogger, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, FORMAT, CHANNELS
from google import genai

# Import CRM API
from crm_api import crm_router
from crm_database import init_database, get_session, Lead, CallSession

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

# Include CRM API routes
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

def create_voice_config(language_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create dynamic voice configuration based on detected language.
    """
    lang_name = language_info['lang']
    formal_address = language_info['formal_address']
    
    # Create system instruction in the detected language
    if lang_name == 'English':
        system_text = f"You are a highly persuasive sales representative for ArtroFlex joint pain relief cream, answering phone calls in {lang_name}. The caller has been greeted, so respond directly to their questions BUT ALWAYS steer the conversation back to ArtroFlex and closing the sale. You are PERSISTENT and URGENT - every conversation is a sales opportunity. Answer their questions briefly, then immediately connect it to ArtroFlex benefits. You specialize in joint health, arthritis treatment, and natural pain relief. Emphasize these key points REPEATEDLY: ArtroFlex's natural ingredients (glucosamine, chondroitin, MSM, turmeric, menthol) provide FAST pain relief in 15 minutes, restore joint flexibility, reduce inflammation with NO side effects. Create URGENCY - mention limited-time offers, special pricing ending soon, or limited stock. Handle every objection as a buying signal: 'I understand your concern about price - that's exactly why we're offering 50% off TODAY ONLY, but I can only hold this price for the next few minutes.' Use assumptive close techniques: 'When would you like to start feeling pain-free - this week or next week?' Don't accept 'no' easily - ask probing questions about their pain, then show how ArtroFlex solves it. Offer special deals: 'Buy 2 get 1 free TODAY', '30-day guarantee', 'Free shipping if you order now'. Always try to close: 'Can I put you down for the 3-bottle package?' Keep responses conversational but ALWAYS with sales intent. Use formal address ({formal_address}). Speak clearly, enthusiastically, and with confidence."
    elif lang_name == 'Bulgarian':
        system_text = f"Вие сте изключително убедителен търговски представител на АртроФлекс - крем за болки в ставите, който отговаря на телефонни обаждания на {lang_name}. Обаждащият е поздравен, отговорете на въпросите му, НО ВИНАГИ насочвайте разговора към АртроФлекс и затварянето на продажбата. Вие сте НАСТОЯТЕЛЕН и НЕОТЛОЖЕН - всеки разговор е продажбена възможност. Отговаряйте кратко на въпросите им, но ВЕДНАГА ги свързвайте с ползите на АртроФлекс. Специализирате се в здравето на ставите, лечението на артрит и естественото облекчаване на болката. Подчертавайте МНОГОКРАТНО тези ключови точки: естествените съставки на АртроФлекс (глюкозамин, хондроитин, MSM, куркума, ментол) осигуряват БЪРЗО облекчаване на болката за 15 минути, възстановяват гъвкавостта на ставите, намаляват възпалението БЕЗ странични ефекти. Създавайте НЕОТЛОЖНОСТ - споменавайте ограничени по време оферти, специални цени, които свършват скоро, или ограничен запас. Справяйте се с всяко възражение като сигнал за купуване: 'Разбирам загриженостите {formal_address} за цената - точно затова предлагаме 50% отстъпка САМО ДНЕС, но мога да задържа тази цена само за следващите няколко минути.' Използвайте техники за предполагащо затваряне: 'Кога би искал да започне да живее без болка - тази седмица или следващата?' Не приемайте 'не' лесно - задавайте проникновени въпроси за болката им, след което покажете как АртроФлекс я решава. Предлагайте специални сделки: 'Купи 2, вземи 1 безплатно ДНЕС', 'Гаранция 30 дни', 'Безплатна доставка при поръчка сега'. Винаги се опитвайте да затворите: 'Мога ли да {formal_address} запиша за пакета от 3 бутилки?' Поддържайте отговорите разговорни, но ВИНАГИ с продажбена цел. Използвайте формално обръщение ({formal_address}). Говорете ясно, ентусиазирано и с увереност."
    else:
        system_text = f"You are a highly persuasive sales representative for ArtroFlex joint pain relief cream, answering phone calls in {lang_name}. The caller has been greeted, so respond directly to their questions BUT ALWAYS steer the conversation back to ArtroFlex and closing the sale. You are PERSISTENT and URGENT - every conversation is a sales opportunity. Answer their questions briefly, then immediately connect it to ArtroFlex benefits. You specialize in joint health, arthritis treatment, and natural pain relief. Emphasize these key points REPEATEDLY: ArtroFlex's natural ingredients (glucosamine, chondroitin, MSM, turmeric, menthol) provide FAST pain relief in 15 minutes, restore joint flexibility, reduce inflammation with NO side effects. Create URGENCY - mention limited-time offers, special pricing ending soon, or limited stock. Handle every objection as a buying signal: 'I understand your concern about price - that's exactly why we're offering 50% off TODAY ONLY, but I can only hold this price for the next few minutes.' Use assumptive close techniques: 'When would you like to start feeling pain-free - this week or next week?' Don't accept 'no' easily - ask probing questions about their pain, then show how ArtroFlex solves it. Offer special deals: 'Buy 2 get 1 free TODAY', '30-day guarantee', 'Free shipping if you order now'. Always try to close: 'Can I put you down for the 3-bottle package?' Keep responses conversational but ALWAYS with sales intent. Use formal address ({formal_address}). Speak clearly, enthusiastically, and with confidence."
    
    return {
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
                    "text": system_text
                }
            ]
        }
    }

# Default voice config (Bulgarian) - will be overridden dynamically
DEFAULT_VOICE_CONFIG = create_voice_config(get_language_config('BG'))
MODEL = "models/gemini-2.0-flash-live-001"

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
                if self.transcription_method == "faster_whisper":
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
            if self.transcription_method == "openai_api":
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
    """Records call audio to WAV files - separate files for incoming and outgoing audio"""
    
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
            incoming_array = np.frombuffer(incoming_audio, dtype=np.int16) if incoming_audio else np.array([], dtype=np.int16)
            outgoing_array = np.frombuffer(outgoing_audio, dtype=np.int16) if outgoing_audio else np.array([], dtype=np.int16)
            
            # Pad shorter array with zeros to match lengths
            max_length = max(len(incoming_array), len(outgoing_array))
            if len(incoming_array) < max_length:
                incoming_array = np.pad(incoming_array, (0, max_length - len(incoming_array)))
            if len(outgoing_array) < max_length:
                outgoing_array = np.pad(outgoing_array, (0, max_length - len(outgoing_array)))
            
            # Mix audio (average to avoid clipping)
            if max_length > 0:
                mixed_array = (incoming_array.astype(np.int32) + outgoing_array.astype(np.int32)) // 2
                mixed_array = np.clip(mixed_array, -32768, 32767).astype(np.int16)
                
                # Save mixed audio
                with wave.open(str(self.mixed_wav_path), 'wb') as mixed_wav:
                    mixed_wav.setnchannels(1)  # Mono
                    mixed_wav.setsampwidth(2)  # 16-bit
                    mixed_wav.setframerate(8000)  # 8kHz
                    mixed_wav.writeframes(mixed_array.tobytes())
                
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
            
        try:
            logger.info(f"🛑 Stopping recording for session {self.session_id}")
            
            # Close WAV files
            if self.incoming_wav:
                self.incoming_wav.close()
                self.incoming_wav = None
                
            if self.outgoing_wav:
                self.outgoing_wav.close()
                self.outgoing_wav = None
            
            # Create mixed recording
            self.create_mixed_recording()
            
            # Create session info file
            self._save_session_info()
            
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
            
            # Start transcription in background thread to avoid blocking (only if available)
            if TRANSCRIPTION_AVAILABLE:
                logger.info(f"🎙️ Starting audio transcription for session {self.session_id}...")
                threading.Thread(target=self._transcribe_recordings, daemon=True).start()
            else:
                logger.info(f"⚠️ Skipping transcription for session {self.session_id} - transcription not available")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _transcribe_recordings(self):
        """Transcribe call recordings using Whisper (runs in background thread)"""
        try:
            logger.info(f"🎤 Transcribing recordings for session {self.session_id}")
            
            # Detect language from caller ID for better transcription
            caller_country = detect_caller_country(self.caller_id)
            language_info = get_language_config(caller_country)
            language_hint = None
            
            # Map language codes to Whisper-compatible codes
            lang_code = language_info.get('code', 'en')
            if lang_code.startswith('en'):
                language_hint = 'en'
            elif lang_code.startswith('bg'):
                language_hint = 'bg'
            elif lang_code.startswith('ro'):
                language_hint = 'ro'
            elif lang_code.startswith('el'):
                language_hint = 'el'
            elif lang_code.startswith('de'):
                language_hint = 'de'
            elif lang_code.startswith('fr'):
                language_hint = 'fr'
            elif lang_code.startswith('es'):
                language_hint = 'es'
            elif lang_code.startswith('it'):
                language_hint = 'it'
            elif lang_code.startswith('ru'):
                language_hint = 'ru'
            # Add more language mappings as needed
            
            logger.info(f"Using language hint for transcription: {language_hint} (from {language_info.get('lang', 'Unknown')})")
            
            # Transcribe all recordings
            transcripts = audio_transcriber.transcribe_call_recordings(
                self.session_dir, 
                language_hint=language_hint
            )
            
            # Update session info with transcript information
            self._update_session_info_with_transcripts(transcripts)
            
            # Log completion
            logger.info(f"✅ Transcription completed for session {self.session_id}")
            for audio_type, result in transcripts.items():
                if result.get('success'):
                    text_length = len(result.get('text', ''))
                    detected_lang = result.get('language', 'unknown')
                    logger.info(f"   {audio_type}: {text_length} characters, language: {detected_lang}")
                else:
                    error = result.get('error', 'Unknown error')
                    logger.error(f"   {audio_type}: Failed - {error}")
            
        except Exception as e:
            logger.error(f"❌ Error transcribing recordings for session {self.session_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_session_info_with_transcripts(self, transcripts: dict):
        """Update session_info.json with transcript information"""
        try:
            info_path = self.session_dir / "session_info.json"
            
            # Load existing session info
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
            else:
                # Create basic session info if it doesn't exist
                session_info = {
                    "session_id": self.session_id,
                    "caller_id": self.caller_id,
                    "called_number": self.called_number,
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat(),
                }
            
            # Add transcript information
            session_info["transcripts"] = {}
            for audio_type, result in transcripts.items():
                session_info["transcripts"][audio_type] = {
                    "success": result.get('success', False),
                    "language": result.get('language', 'unknown'),
                    "text_length": len(result.get('text', '')),
                    "confidence": result.get('confidence'),
                    "transcript_file": result.get('transcript_file'),
                    "transcribed_at": datetime.now(timezone.utc).isoformat()
                }
                
                if not result.get('success'):
                    session_info["transcripts"][audio_type]["error"] = result.get('error')
            
            # Save updated session info
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2, ensure_ascii=False)
                
            logger.info(f"💾 Updated session info with transcript data: {info_path}")
            
        except Exception as e:
            logger.error(f"❌ Error updating session info with transcripts: {e}")
    
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
    
    def create_session(self, session_id: str, remote_addr, voice_session, call_recorder=None):
        """Create RTP session for a call"""
        rtp_session = RTPSession(session_id, remote_addr, self.socket, voice_session, call_recorder)
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
    
    def __init__(self, session_id: str, remote_addr, rtp_socket, voice_session, call_recorder=None):
        self.session_id = session_id
        self.remote_addr = remote_addr
        self.rtp_socket = rtp_socket
        self.voice_session = voice_session
        self.call_recorder = call_recorder  # Add call recorder
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = hash(session_id) & 0xFFFFFFFF
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.input_processing = False  # For incoming audio processing
        
        # Output audio queue for paced delivery
        self.output_queue = queue.Queue()
        self.output_thread = None
        self.output_processing = True  # For outgoing audio processing
        
        # Audio level tracking for AGC
        self.audio_level_history = []
        self.target_audio_level = -20  # dBFS
        
        # Crossfade buffer for smooth transitions
        self.last_audio_tail = b""  # Last 10ms of previous chunk
        
        # Initialize audio preprocessor for incoming audio
        self.audio_preprocessor = AudioPreprocessor(sample_rate=8000)
        logger.info(f"🎵 Audio preprocessor initialized for session {session_id} - noise filtering enabled")
        
        # Start output processing thread immediately for greeting playback
        self.output_thread = threading.Thread(target=self._process_output_queue, daemon=True)
        self.output_thread.start()
        logger.info(f"🎵 Started output queue processing for session {session_id}")
        
        # Keep processing flag for backward compatibility with other parts of the code
        self.processing = self.output_processing
        
    def process_incoming_audio(self, audio_data: bytes, payload_type: int, timestamp: int):
        """Process incoming RTP audio packet with noise filtering"""
        try:
            # Convert payload based on type
            if payload_type == 0:  # PCMU/μ-law
                pcm_data = self.ulaw_to_pcm(audio_data)
            elif payload_type == 8:  # PCMA/A-law  
                pcm_data = self.alaw_to_pcm(audio_data)
            else:
                # Assume it's already PCM
                pcm_data = audio_data
            
            # Apply audio preprocessing for noise reduction and speech enhancement
            try:
                processed_pcm = self.audio_preprocessor.process_audio(pcm_data)
                logger.debug(f"🎵 Audio preprocessed: {len(pcm_data)} → {len(processed_pcm)} bytes")
            except Exception as preprocess_error:
                logger.warning(f"⚠️ Audio preprocessing failed, using original: {preprocess_error}")
                processed_pcm = pcm_data
            
            # Record the ORIGINAL audio (before preprocessing) for authentic recordings
            if self.call_recorder:
                self.call_recorder.record_incoming_audio(pcm_data)
            
            # Queue the PROCESSED audio for AI transcription and processing
            self.audio_queue.put(processed_pcm)
            logger.info(f"🎤 Queued {len(processed_pcm)} bytes of enhanced PCM audio for AI processing")
            
            # Start input processing if not already running
            if not self.input_processing:
                self.input_processing = True
                threading.Thread(target=self._process_audio_queue, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
    
    def _process_audio_queue(self):
        """Process queued audio through voice session optimized for maximum performance"""
        audio_buffer = b""
        chunk_size = 480  # 30ms at 8kHz for maximum processing stability
        
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
            self.input_processing = False
    
    async def _async_process_audio_queue(self, audio_buffer: bytes, chunk_size: int):
        """Async version of audio queue processing optimized for maximum performance and stability"""
        try:
            # Initialize voice session first
            if not await self.voice_session.initialize_voice_session():
                logger.error("Failed to initialize voice session")
                return
                
            # Start receive task for continuous response handling
            receive_task = asyncio.create_task(self._continuous_receive_responses())
            
            # Large chunk sizes for maximum performance and stability
            min_chunk_size = 480  # 30ms minimum - high performance
            max_chunk_size = 640  # 40ms maximum - very stable processing
            
            while self.input_processing:
                try:
                    # Get audio chunk immediately without any timeout
                    try:
                        audio_chunk = self.audio_queue.get_nowait()
                        audio_buffer += audio_chunk
                    except queue.Empty:
                        # Send any buffered audio immediately, no matter how small
                        if len(audio_buffer) >= min_chunk_size and self.voice_session.gemini_session:
                            await self._send_audio_to_gemini(audio_buffer)
                            audio_buffer = b""
                        # Yield control less frequently with larger chunks
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Send audio immediately as soon as we have minimum data
                    while len(audio_buffer) >= min_chunk_size:
                        chunk_to_process = audio_buffer[:max_chunk_size]
                        audio_buffer = audio_buffer[max_chunk_size:]
                        
                        # Send audio to Gemini immediately without any delays
                        if self.voice_session.gemini_session:
                            await self._send_audio_to_gemini(chunk_to_process)
                        
                except Exception as e:
                    logger.error(f"Error in async audio processing: {e}")
                    # No retry delay - continue immediately
                    continue
                    
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
                input={"data": processed_audio, "mime_type": "audio/pcm;rate=16000"}
            )
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")
            
    async def _continuous_receive_responses(self):
        """Continuously receive responses from Gemini in a separate task"""
        logger.info("🎧 Starting continuous response receiver")
        
        while self.input_processing and self.voice_session.gemini_session:
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
                                                    
                                                    # Send immediately in small chunks for lowest latency
                                                    self._send_audio_immediate(telephony_audio)
                                                except Exception as e:
                                                    logger.error(f"Error decoding base64 audio: {e}")
                                            elif isinstance(audio_data, bytes):
                                                logger.info(f"📥 Received {len(audio_data)} bytes of audio from Gemini")
                                                # Convert and send in smaller chunks to avoid overwhelming the receiver
                                                telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_data)
                                                
                                                # Send immediately in small chunks for lowest latency
                                                self._send_audio_immediate(telephony_audio)
                                    
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
                                # Send immediately in small chunks for lowest latency
                                self._send_audio_immediate(telephony_audio)
                            elif isinstance(response.data, str):
                                try:
                                    audio_bytes = base64.b64decode(response.data)
                                    logger.info(f"📥 Received {len(audio_bytes)} bytes of audio from Gemini (direct base64)")
                                    telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_bytes)
                                    # Send immediately in small chunks for lowest latency
                                    self._send_audio_immediate(telephony_audio)
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
                if self.input_processing:  # Only log if we're still processing
                    logger.error(f"Error receiving from Gemini: {e}")
                    # No sleep delay - continue immediately for fastest recovery
                    
        logger.info("🎧 Stopped continuous response receiver")
    
    def _send_audio_immediate(self, audio_data: bytes):
        """Send audio data immediately in ultra-small chunks for lowest latency"""
        if not audio_data:
            return
            
        # Send in large chunks (30ms each) for maximum performance
        chunk_size = 480  # 30ms at 8kHz = 240 samples * 2 bytes = 480 bytes
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            # Send each chunk immediately to the output queue
            self.send_audio(chunk)
    
    # Removed _process_chunk - we use continuous streaming instead
    
    def send_audio(self, pcm16: bytes):
        """
        Takes 16-bit mono PCM at 8000 Hz, μ-law encodes, then enqueues in ultra-small packets for low latency.
        """
        # Record outgoing audio before encoding
        if self.call_recorder:
            self.call_recorder.record_outgoing_audio(pcm16)
        
        ulaw = self.pcm_to_ulaw(pcm16)  # your existing encoder
        packet_size = 240  # 30ms @ 8000 Hz, 1 byte/sample for G.711 - maximum stability
        for i in range(0, len(ulaw), packet_size):
            self.output_queue.put(ulaw[i:i + packet_size])
    
    def play_greeting_file(self, greeting_file: str = "greeting.wav"):
        """Play a greeting WAV file at the start of the call and return estimated duration"""
        try:
            if not os.path.exists(greeting_file):
                logger.warning(f"Greeting file {greeting_file} not found, skipping greeting")
                return 0.0
                
            logger.info(f"🎵 Playing greeting file: {greeting_file}")
            
            # Load WAV file
            with wave.open(greeting_file, 'rb') as wav_file:
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
                    from scipy.signal import resample
                    
                    # Convert to numpy array for resampling
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Calculate target sample count
                    target_samples = int(len(audio_array) * 8000 / sample_rate)
                    
                    # Resample
                    resampled_array = resample(audio_array, target_samples)
                    
                    # Convert back to bytes
                    audio_data = resampled_array.astype(np.int16).tobytes()
                    
                    # Update duration for resampled audio
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
        Dequeue G.711 payloads and transmit with ultra-low latency RTP pacing.
        Optimized for immediate audio delivery.
        """
        ptime_ms = 30  # Large 30ms chunks for maximum stability
        frame_bytes = 240  # 30ms of G.711 for high performance
        
        while self.output_processing:
            try:
                payload = self.output_queue.get(timeout=0.1)  # Larger timeout for stability
            except Exception:
                # No artificial delays or comfort noise - stay completely silent
                continue

            # Transmit immediately with minimal pacing
            self._send_rtp(payload)
            self._sleep_ms(ptime_ms)
    
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
        """Convert 16-bit PCM to μ-law with ultra-light soft limiting"""
        try:
            # Apply ultra-light soft limiting to prevent μ-law harshness on peaks
            pcm_data = self.soft_limit_16bit(pcm_data, 0.95)
            
            # Convert to μ-law
            return audioop.lin2ulaw(pcm_data, 2)
        except Exception as e:
            logger.error(f"Error in PCM to μ-law conversion: {e}")
            # Fallback to simple conversion
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
            logger.info(f"📞 Raw caller info for language detection: {caller_id}")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Send 180 Ringing first
            logger.info("📤 Sending 180 Ringing...")
            ringing_response = self._create_sip_ringing_response(message, session_id)
            self.socket.sendto(ringing_response.encode(), addr)
            logger.info("🔔 180 Ringing sent")
            
            # Send 200 OK immediately - no delay needed
            
            # Send 200 OK response
            logger.info("📤 Sending 200 OK response...")
            ok_response = self._create_sip_ok_response(message, session_id)
            logger.debug(f"200 OK response:\n{ok_response}")
            self.socket.sendto(ok_response.encode(), addr)
            logger.info("✅ 200 OK response sent")
            
            # Create or update CRM database - call answered
            db = get_session()
            try:
                crm_session = db.query(CallSession).filter(CallSession.session_id == session_id).first()
                if crm_session:
                    crm_session.status = "answered"
                else:
                    # Create new session for incoming calls
                    crm_session = CallSession(
                        session_id=session_id,
                        caller_id=caller_id,
                        called_number=config.get('phone_number', 'Unknown'),
                        status="answered",
                        started_at=datetime.utcnow(),
                        answered_at=datetime.utcnow()
                    )
                    db.add(crm_session)
                db.commit()
            except Exception as e:
                logger.error(f"Error updating CRM session status: {e}")
            finally:
                db.close()
            
            # Detect caller's country and language
            logger.info(f"🌍 ======== LANGUAGE DETECTION ========")
            logger.info(f"📞 Caller ID: {caller_id}")
            
            caller_country = detect_caller_country(caller_id)
            language_info = get_language_config(caller_country)
            voice_config = create_voice_config(language_info)
            
            logger.info(f"🌍 ✅ Detected country: {caller_country}")
            logger.info(f"🗣️ ✅ Selected language: {language_info['lang']} ({language_info['code']})")
            logger.info(f"👤 ✅ Formal address: {language_info['formal_address']}")
            logger.info(f"🌍 =====================================")
            
            # Create voice session with language-specific configuration
            logger.info(f"🎙️  Creating voice session {session_id} with {language_info['lang']} language config")
            voice_session = WindowsVoiceSession(session_id, caller_id, self.phone_number, voice_config)
            
            # Create RTP session for audio
            rtp_session = self.rtp_server.create_session(session_id, addr, voice_session, voice_session.call_recorder)
            
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
                        
                        # Start AI conversation immediately without waiting for user input
                        def start_ai_conversation():
                            time.sleep(0.5)  # Allow RTP connection to stabilize
                            logger.info("🎵 Playing greeting and triggering AI response...")
                            
                            # Play greeting file first and get actual duration
                            greeting_duration = rtp_session.play_greeting_file("greeting.wav")
                            
                            # Wait for greeting to finish playing plus small buffer
                            wait_time = max(greeting_duration + 0.5, 2.0)  # At least 2 seconds, or duration + buffer
                            logger.info(f"⏱️ Waiting {wait_time:.1f}s for greeting to complete...")
                            time.sleep(wait_time)
                            
                            # Now trigger AI to start conversation by sending a small audio trigger
                            if session_id in active_sessions and active_sessions[session_id]["voice_session"]:
                                voice_session = active_sessions[session_id]["voice_session"]
                                if hasattr(voice_session, 'gemini_session') and voice_session.gemini_session:
                                    try:
                                        # Send a small audio trigger to start AI conversation
                                        # Use a very brief audio sample to trigger response
                                        trigger_audio = b'\x00\x01' * 160  # 20ms of minimal audio at 8kHz
                                        processed_trigger = voice_session.convert_telephony_to_gemini(trigger_audio)
                                        
                                        # Create async function to trigger AI conversation start
                                        async def trigger_ai_conversation():
                                            try:
                                                await voice_session.gemini_session.send(
                                                    input={"data": processed_trigger, "mime_type": "audio/pcm;rate=16000"}
                                                )
                                                logger.info("✅ AI conversation triggered after greeting")
                                            except Exception as e:
                                                logger.error(f"Error triggering AI conversation: {e}")
                                        
                                        # Run the trigger in a new event loop
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        try:
                                            loop.run_until_complete(trigger_ai_conversation())
                                        finally:
                                            loop.close()
                                            
                                    except Exception as e:
                                        logger.error(f"Error setting up AI conversation trigger: {e}")
                        
                        threading.Thread(target=start_ai_conversation, daemon=True).start()
                        
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
                    
                    # Update CRM database
                    db = get_session()
                    try:
                        crm_session = db.query(CallSession).filter(CallSession.session_id == session_id).first()
                        if crm_session:
                            crm_session.status = "COMPLETED"  # Use proper enum value
                            crm_session.ended_at = datetime.utcnow()
                            if crm_session.started_at and crm_session.ended_at:
                                crm_session.duration = int((crm_session.ended_at - crm_session.started_at).total_seconds())
                            db.commit()
                    except Exception as e:
                        logger.error(f"Error updating CRM session: {e}")
                    finally:
                        db.close()
                    
                    # Cleanup voice session
                    voice_session = session_data["voice_session"]
                    rtp_session = session_data.get("rtp_session")
                    
                    # Stop RTP processing first
                    if rtp_session:
                        rtp_session.input_processing = False
                        rtp_session.output_processing = False
                        rtp_session.processing = False  # For backward compatibility
                    
                    try:
                        if hasattr(voice_session, 'cleanup'):
                            # Handle cleanup properly without creating async task in wrong context
                            voice_session.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up voice session: {e}")
                    
                    # Cleanup RTP session
                    self.rtp_server.remove_session(session_id)
                    
                    # Small delay to allow pending tasks to complete
                    try:
                        import time
                        time.sleep(0.1)  # Give pending tasks a moment to complete
                    except:
                        pass
                    
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
    
    def make_outbound_call(self, phone_number: str) -> Optional[str]:
        """Initiate outbound call as registered Extension 200"""
        try:
            # Check if extension is registered
            if not self.extension_registered:
                logger.error("❌ Cannot make outbound call - Extension 200 not registered")
                logger.error("💡 Wait for extension registration to complete")
                return None
            
            logger.info(f"📞 Initiating outbound call to {phone_number} as Extension 200")
            logger.info(f"📞 Call will be routed through Gate VoIP → {self.config.get('outbound_trunk', 'gsm2')} trunk")
            
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
            
            # Create INVITE as Extension 200
            invite_message = f"""INVITE sip:{phone_number}@{self.gate_ip} SIP/2.0
Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{session_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{self.gate_ip}>;tag={session_id[:8]}
To: <sip:{phone_number}@{self.gate_ip}>
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
                'phone_number': phone_number,
                'session_id': session_id,
                'username': username,
                'sdp_content': sdp_content,
                'cseq': 1
            }
            
            # Send INVITE
            self.socket.sendto(invite_message.encode(), (self.gate_ip, self.sip_port))
            logger.info(f"📤 Sent INVITE for outbound call to {phone_number}")
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
    
    def _handle_outbound_call_success(self, message: str):
        """Handle successful outbound call establishment (200 OK for INVITE)"""
        try:
            # Extract Call-ID to find the pending invite
            call_id = None
            to_tag = None
            from_tag = None
            
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
                elif line.startswith('From:') and 'tag=' in line:
                    tag_start = line.find('tag=') + 4
                    tag_end = line.find(';', tag_start)
                    if tag_end == -1:
                        tag_end = line.find('>', tag_start)
                    if tag_end == -1:
                        tag_end = len(line)
                    from_tag = line[tag_start:tag_end].strip()
            
            if not call_id or call_id not in self.pending_invites:
                logger.warning("❌ Cannot find pending INVITE for successful response")
                return
            
            invite_info = self.pending_invites[call_id]
            phone_number = invite_info['phone_number']
            session_id = invite_info['session_id']
            
            logger.info(f"✅ Outbound call to {phone_number} answered successfully!")
            logger.info(f"📞 Call established - Session ID: {session_id}")
            
            # Detect caller's country and language (for outbound calls, we use our own number)
            caller_country = detect_caller_country(self.phone_number)  # Use our own number
            language_info = get_language_config(caller_country)
            voice_config = create_voice_config(language_info)
            
            logger.info(f"🗣️ Using {language_info['lang']} for outbound call")
            
            # Create voice session for outbound call (we are the caller)
            voice_session = WindowsVoiceSession(
                session_id, 
                self.phone_number,  # We are calling
                phone_number,       # They are receiving
                voice_config
            )
            
            # Create RTP session - for outbound calls, we need to get the remote address from SDP
            # For now, we'll use the Gate VoIP address as the RTP destination
            remote_addr = (self.gate_ip, 5004)  # Default RTP port
            rtp_session = self.rtp_server.create_session(session_id, remote_addr, voice_session, voice_session.call_recorder)
            
            # Store the active session
            active_sessions[session_id] = {
                "voice_session": voice_session,
                "rtp_session": rtp_session,
                "caller_addr": (self.gate_ip, self.sip_port),
                "status": "connecting",
                "call_start": datetime.now(timezone.utc),
                "call_id": call_id,
                "call_type": "outbound"
            }
            
            # Send ACK to complete the call setup
            self._send_ack(call_id, to_tag, from_tag, invite_info)
            
            # Clean up pending invite
            del self.pending_invites[call_id]
            
            # Start voice session in separate thread
            def start_voice_session():
                try:
                    asyncio.run(voice_session.initialize_voice_session())
                    # Mark as active once voice session is ready
                    if session_id in active_sessions:
                        active_sessions[session_id]["status"] = "active"
                        logger.info(f"🎯 Outbound voice session {session_id} is now active and ready")
                        
                        # For outbound calls, we should start talking first
                        def start_outbound_conversation():
                            time.sleep(0.5)  # Small delay for RTP to stabilize
                            logger.info("🎵 Playing greeting and starting conversation with called party...")
                            
                            # Play greeting file first and get actual duration
                            greeting_duration = rtp_session.play_greeting_file("greeting.wav")
                            
                            # Wait for greeting to finish playing plus small buffer
                            wait_time = max(greeting_duration + 0.5, 2.0)  # At least 2 seconds, or duration + buffer
                            logger.info(f"⏱️ Waiting {wait_time:.1f}s for greeting to complete...")
                            time.sleep(wait_time)
                            
                            # Now trigger AI to start conversation for outbound call
                            if session_id in active_sessions and active_sessions[session_id]["voice_session"]:
                                voice_session = active_sessions[session_id]["voice_session"]
                                if hasattr(voice_session, 'gemini_session') and voice_session.gemini_session:
                                    try:
                                        # Send a small audio trigger to start AI conversation
                                        trigger_audio = b'\x00\x01' * 160  # 20ms of minimal audio at 8kHz
                                        processed_trigger = voice_session.convert_telephony_to_gemini(trigger_audio)
                                        
                                        # Create async function to trigger AI conversation start
                                        async def trigger_outbound_ai_conversation():
                                            try:
                                                await voice_session.gemini_session.send(
                                                    input={"data": processed_trigger, "mime_type": "audio/pcm;rate=16000"}
                                                )
                                                logger.info("✅ AI conversation triggered for outbound call after greeting")
                                            except Exception as e:
                                                logger.error(f"Error triggering outbound AI conversation: {e}")
                                        
                                        # Run the trigger in a new event loop
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        try:
                                            loop.run_until_complete(trigger_outbound_ai_conversation())
                                        finally:
                                            loop.close()
                                            
                                    except Exception as e:
                                        logger.error(f"Error setting up outbound AI conversation trigger: {e}")
                        
                        threading.Thread(target=start_outbound_conversation, daemon=True).start()
                        
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
            logger.info(f"📤 Sent ACK to complete call setup")
            logger.debug(f"ACK message:\n{ack_message}")
            
        except Exception as e:
            logger.error(f"❌ Error sending ACK: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _handle_sip_response(self, message: str, addr):
        """Handle SIP responses (200 OK, 401 Unauthorized, etc.)"""
        try:
            first_line = message.split('\n')[0].strip()
            
            if '200 OK' in first_line:
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
                        logger.error(f"❌ Outbound call to {phone_number} failed: {first_line}")
                        logger.error("💡 This usually means the outgoing route is misconfigured")
                        logger.error("💡 CRITICAL: Check Gate VoIP outgoing route uses GSM2 trunk (not voice-agent trunk)")
                        logger.error("💡 Go to: http://192.168.50.50 > PBX Settings > Outgoing Routes")
                        logger.error("💡 Change trunk from 'voice-agent' to 'gsm2' and save")
                        
                        # Clean up the failed call
                        del self.pending_invites[call_id]
                    else:
                        logger.error(f"❌ Call not found: {first_line}")
                else:
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
    
    def __init__(self, session_id: str, caller_id: str, called_number: str, voice_config: Dict[str, Any] = None):
        self.session_id = session_id
        self.caller_id = caller_id
        self.called_number = called_number
        self.start_time = datetime.now(timezone.utc)
        self.session_logger = SessionLogger()
        self.voice_session = None
        self.gemini_session = None  # The actual session object from the context manager
        
        # Initialize call recorder
        self.call_recorder = CallRecorder(session_id, caller_id, called_number)
        logger.info(f"🎙️ Call recorder initialized for session {session_id}")
        
        # Use provided voice config or default
        self.voice_config = voice_config if voice_config else DEFAULT_VOICE_CONFIG
        
        # Connection management
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.connection_backoff = 1.0  # seconds
        self.last_connection_attempt = 0
        
        # Audio resampling state for fast conversion
        self._to16k_state = None  # 8k -> 16k state
        self._to8k_state = None   # 24k -> 8k state
        
        # Log call start with language info
        system_text = self.voice_config.get('system_instruction', {}).get('parts', [{}])[0].get('text', '')
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
            # Extract language from voice config for logging
            system_text = self.voice_config.get('system_instruction', {}).get('parts', [{}])[0].get('text', '')
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
                context_manager = voice_client.aio.live.connect(
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
                        input={"data": processed_audio, "mime_type": "audio/pcm;rate=16000"}
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
        Input: 16-bit mono PCM at 8000 Hz (after μ-law decode, if applicable).
        Output: 16-bit mono PCM at 16000 Hz for the model.
        """
        # Convert bytes to numpy array
        in_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate the number of samples in the output signal
        num_samples_in = len(in_array)
        num_samples_out = int(num_samples_in * 16000 / 8000)
        
        # Resample using scipy for high quality
        resampled_array = resample(in_array, num_samples_out)
        
        # Convert back to bytes
        return resampled_array.astype(np.int16).tobytes()

    def convert_gemini_to_telephony(self, model_pcm: bytes) -> bytes:
        """
        Input: model PCM as 16-bit mono at 24000 Hz (typical).
        Output: 16-bit mono PCM at 8000 Hz ready for μ-law encode.
        """
        # Convert bytes to numpy array
        in_array = np.frombuffer(model_pcm, dtype=np.int16)
        
        # Calculate the number of samples in the output signal
        num_samples_in = len(in_array)
        num_samples_out = int(num_samples_in * 8000 / 24000)
        
        # Resample using scipy for high quality
        resampled_array = resample(in_array, num_samples_out)
        
        # Convert back to bytes
        return resampled_array.astype(np.int16).tobytes()
    
    
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

# Initialize SIP handler
sip_handler = WindowsSIPHandler()

@app.on_event("startup")
async def startup_event():
    """Start SIP listener when FastAPI starts"""
    # Initialize CRM database
    init_database()
    logger.info("CRM database initialized")
    
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
            # Save to CRM database
            db = get_session()
            try:
                # Find lead by phone number
                lead = db.query(Lead).filter(
                    (Lead.phone == phone_number) |
                    (Lead.phone == phone_number.replace('+', '')) |
                    (Lead.phone.contains(phone_number[-10:]))  # Last 10 digits
                ).first()
                
                # Update lead if found
                if lead:
                    lead.call_count += 1
                    lead.last_called_at = datetime.utcnow()
                
                # Create session record
                new_session = CallSession(
                    session_id=session_id,
                    called_number=phone_number,
                    lead_id=lead.id if lead else None,
                    campaign_id=None,  # Manual call
                    status="dialing",
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
        "transcription_available": TRANSCRIPTION_AVAILABLE,
        "transcription_method": TRANSCRIPTION_METHOD if TRANSCRIPTION_AVAILABLE else None,
        "transcription_model": audio_transcriber.model_size if TRANSCRIPTION_AVAILABLE else None,
        "audio_preprocessing": {
            "enabled": True,
            "features": [
                "Pre-emphasis filtering",
                "Bandpass filtering (300-3400 Hz)",
                "Voice activity detection",
                "Spectral noise reduction",
                "Dynamic range compression",
                "Audio normalization"
            ],
            "purpose": "Enhanced speech recognition for incoming audio"
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
async def get_recordings():
    """Get list of available call recordings"""
    try:
        sessions_dir = Path("sessions")
        recordings = []
        
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    session_info_path = session_dir / "session_info.json"
                    if session_info_path.exists():
                        try:
                            with open(session_info_path, 'r') as f:
                                session_info = json.load(f)
                            
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
                            
                            # Check if transcript files exist
                            transcript_files = {}
                            transcripts_info = session_info.get('transcripts', {})
                            for audio_type, transcript_info in transcripts_info.items():
                                transcript_filename = transcript_info.get('transcript_file')
                                if transcript_filename:
                                    transcript_path = session_dir / transcript_filename
                                    if transcript_path.exists():
                                        transcript_files[audio_type] = {
                                            "filename": transcript_filename,
                                            "language": transcript_info.get('language', 'unknown'),
                                            "text_length": transcript_info.get('text_length', 0),
                                            "confidence": transcript_info.get('confidence'),
                                            "transcribed_at": transcript_info.get('transcribed_at'),
                                            "success": transcript_info.get('success', False),
                                            "path": str(transcript_path)
                                        }
                                        
                                        if not transcript_info.get('success'):
                                            transcript_files[audio_type]["error"] = transcript_info.get('error')
                            
                            if audio_files:  # Only include if audio files exist
                                recording_entry = {
                                    "session_id": session_info.get('session_id'),
                                    "caller_id": session_info.get('caller_id'),
                                    "called_number": session_info.get('called_number'),
                                    "start_time": session_info.get('start_time'),
                                    "end_time": session_info.get('end_time'),
                                    "duration_seconds": session_info.get('duration_seconds'),
                                    "audio_files": audio_files,
                                    "transcript_files": transcript_files,
                                    "has_transcripts": len(transcript_files) > 0
                                }
                                recordings.append(recording_entry)
                        except Exception as e:
                            logger.warning(f"Error reading session info for {session_dir.name}: {e}")
                            continue
        
        # Sort by start_time, newest first
        recordings.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return {
            "status": "success",
            "total_recordings": len(recordings),
            "recordings": recordings
        }
        
    except Exception as e:
        logger.error(f"Error getting recordings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcripts/{session_id}")
async def get_session_transcripts(session_id: str):
    """Get transcripts for a specific session"""
    try:
        session_dir = Path(f"sessions/{session_id}")
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Load session info
        session_info_path = session_dir / "session_info.json"
        if not session_info_path.exists():
            raise HTTPException(status_code=404, detail="Session info not found")
        
        with open(session_info_path, 'r', encoding='utf-8') as f:
            session_info = json.load(f)
        
        # Get transcript information
        transcripts_info = session_info.get('transcripts', {})
        transcripts = {}
        
        for audio_type, transcript_info in transcripts_info.items():
            transcript_filename = transcript_info.get('transcript_file')
            if transcript_filename:
                transcript_path = session_dir / transcript_filename
                if transcript_path.exists():
                    # Read transcript content
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript_content = f.read()
                    
                    transcripts[audio_type] = {
                        "filename": transcript_filename,
                        "language": transcript_info.get('language', 'unknown'),
                        "text_length": transcript_info.get('text_length', 0),
                        "confidence": transcript_info.get('confidence'),
                        "transcribed_at": transcript_info.get('transcribed_at'),
                        "success": transcript_info.get('success', False),
                        "content": transcript_content
                    }
                    
                    if not transcript_info.get('success'):
                        transcripts[audio_type]["error"] = transcript_info.get('error')
        
        return {
            "status": "success",
            "session_id": session_id,
            "session_info": {
                "caller_id": session_info.get('caller_id'),
                "called_number": session_info.get('called_number'),
                "start_time": session_info.get('start_time'),
                "end_time": session_info.get('end_time'),
                "duration_seconds": session_info.get('duration_seconds')
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
            raise HTTPException(status_code=404, detail="Session not found")
        
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
            "audio_files_found": len(audio_files_found)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting retranscription for session {session_id}: {e}")
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

async def _background_transcribe_session(session_dir: Path, caller_id: str):
    """Background task function for transcribing a session"""
    try:
        logger.info(f"🎤 Background transcription started for {session_dir.name}")
        
        # Detect language from caller ID
        caller_country = detect_caller_country(caller_id)
        language_info = get_language_config(caller_country)
        language_hint = None
        
        # Map language codes to Whisper-compatible codes
        lang_code = language_info.get('code', 'en')
        if lang_code.startswith('en'):
            language_hint = 'en'
        elif lang_code.startswith('bg'):
            language_hint = 'bg'
        elif lang_code.startswith('ro'):
            language_hint = 'ro'
        elif lang_code.startswith('el'):
            language_hint = 'el'
        elif lang_code.startswith('de'):
            language_hint = 'de'
        elif lang_code.startswith('fr'):
            language_hint = 'fr'
        elif lang_code.startswith('es'):
            language_hint = 'es'
        elif lang_code.startswith('it'):
            language_hint = 'it'
        elif lang_code.startswith('ru'):
            language_hint = 'ru'
        
        logger.info(f"Using language hint: {language_hint} (from {language_info.get('lang', 'Unknown')})")
        
        # Transcribe all recordings
        transcripts = audio_transcriber.transcribe_call_recordings(
            session_dir, 
            language_hint=language_hint
        )
        
        # Update session info with transcript information
        info_path = session_dir / "session_info.json"
        
        # Load existing session info
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
        else:
            session_info = {}
        
        # Add transcript information
        session_info["transcripts"] = {}
        for audio_type, result in transcripts.items():
            session_info["transcripts"][audio_type] = {
                "success": result.get('success', False),
                "language": result.get('language', 'unknown'),
                "text_length": len(result.get('text', '')),
                "confidence": result.get('confidence'),
                "transcript_file": result.get('transcript_file'),
                "transcribed_at": datetime.now(timezone.utc).isoformat()
            }
            
            if not result.get('success'):
                session_info["transcripts"][audio_type]["error"] = result.get('error')
        
        # Save updated session info
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Background transcription completed for {session_dir.name}")
        
    except Exception as e:
        logger.error(f"❌ Error in background transcription for {session_dir.name}: {e}")
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
    """Get audio preprocessing status and configuration"""
    return {
        "status": "success",
        "preprocessing_enabled": True,
        "sample_rate": 8000,
        "processing_pipeline": [
            {
                "step": 1,
                "name": "Pre-emphasis filtering",
                "description": "Enhances high-frequency components important for speech",
                "coefficient": 0.97
            },
            {
                "step": 2,
                "name": "Bandpass filtering",
                "description": "Focuses on telephony speech frequencies",
                "frequency_range": "300-3400 Hz"
            },
            {
                "step": 3,
                "name": "Voice activity detection",
                "description": "Identifies and gates background noise during silence",
                "adaptive_threshold": True
            },
            {
                "step": 4,
                "name": "Spectral noise reduction",
                "description": "Removes background noise using spectral subtraction",
                "method": "Adaptive spectral subtraction"
            },
            {
                "step": 5,
                "name": "Dynamic range compression",
                "description": "Evens out volume levels for consistent speech",
                "threshold": 0.7,
                "ratio": "4:1"
            },
            {
                "step": 6,
                "name": "Audio normalization",
                "description": "Normalizes audio levels while preserving dynamics",
                "target_rms": -20  # dBFS
            }
        ],
        "benefits": [
            "Improved speech-to-text transcription accuracy",
            "Reduced background noise interference", 
            "Better voice activity detection",
            "Consistent audio levels",
            "Enhanced speech clarity"
        ],
        "note": "Preprocessing only applied to incoming audio (user speech), not outgoing AI responses"
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
    if TRANSCRIPTION_AVAILABLE:
        logger.info(f"🎤 Transcription: {TRANSCRIPTION_METHOD} with '{audio_transcriber.model_size}' model")
        logger.info("🇧🇬 Optimized for Bulgarian and multilingual transcription")
    else:
        logger.info("⚠️ Transcription: DISABLED (no method available)")
    logger.info("🎵 Audio Preprocessing: ENABLED (6-stage noise filtering pipeline)")
    logger.info("   ✅ Pre-emphasis, bandpass filtering, noise reduction, compression")
    logger.info("   🎯 Optimized for improved user speech transcription accuracy")
    logger.info("🔊 Greeting System: ENABLED for both incoming and outbound calls")
    logger.info("   🎙️ Plays greeting.wav file automatically when call is answered")
    logger.info("   🤖 Triggers AI conversation after greeting completes")
    logger.info("   📏 Dynamic timing based on actual audio file duration")
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
    
    uvicorn.run(
        "windows_voice_agent:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
