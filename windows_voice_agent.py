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

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import pyaudio
import requests
from scipy.signal import resample
import numpy as np

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
        system_text = f"You are an expert affiliate marketing consultant answering phone calls in {lang_name}. The caller has already been greeted, so respond directly to their questions or requests. You specialize in affiliate marketing strategies, lead generation, FTDs (First Time Deposits), CPA networks, affiliate networks, commission structures, traffic generation, conversion optimization, and building successful affiliate campaigns. You have deep knowledge of performance marketing, lead qualification, deposit optimization for trading/forex/fintech offers, working with affiliate networks like MaxBounty, ClickBank, ShareASale, and CPA networks. Share actionable insights about choosing profitable verticals, generating high-quality leads, optimizing FTD rates, building audiences, media buying, and scaling affiliate income. Always respond in {lang_name} and maintain a professional but enthusiastic tone about affiliate marketing opportunities. Keep responses conversational and concise since this is a voice-only interaction. Speak clearly with good enunciation, at a moderate pace, and avoid mumbling or speaking too softly. Project your voice as if speaking over a phone line. Use formal address ({formal_address}) when speaking to callers."
    elif lang_name == 'Bulgarian':
        system_text = f"You are an expert affiliate marketing consultant answering phone calls in {lang_name}. The caller has already been greeted, so respond directly to their questions or requests. You specialize in affiliate marketing strategies, lead generation, FTDs (First Time Deposits), CPA networks, affiliate networks, commission structures, traffic generation, conversion optimization, and building successful affiliate campaigns. You have deep knowledge of performance marketing, lead qualification, deposit optimization for trading/forex/fintech offers, working with affiliate networks like MaxBounty, ClickBank, ShareASale, and CPA networks. Share actionable insights about choosing profitable verticals, generating high-quality leads, optimizing FTD rates, building audiences, media buying, and scaling affiliate income. Always respond in {lang_name} language and maintain a professional but enthusiastic tone about affiliate marketing opportunities. Keep responses conversational and concise since this is a voice-only interaction. Speak clearly with good enunciation, at a moderate pace, and avoid mumbling or speaking too softly. Project your voice as if speaking over a phone line. Use formal {lang_name} address ({formal_address}) when speaking to callers."
    else:
        system_text = f"You are an expert affiliate marketing consultant answering phone calls in {lang_name}. The caller has already been greeted, so respond directly to their questions or requests. You specialize in affiliate marketing strategies, lead generation, FTDs (First Time Deposits), CPA networks, affiliate networks, commission structures, traffic generation, conversion optimization, and building successful affiliate campaigns. You have deep knowledge of performance marketing, lead qualification, deposit optimization for trading/forex/fintech offers, working with affiliate networks like MaxBounty, ClickBank, ShareASale, and CPA networks. Share actionable insights about choosing profitable verticals, generating high-quality leads, optimizing FTD rates, building audiences, media buying, and scaling affiliate income. Always respond in {lang_name} language and maintain a professional but enthusiastic tone about affiliate marketing opportunities. Keep responses conversational and concise since this is a voice-only interaction. Speak clearly with good enunciation, at a moderate pace, and avoid mumbling or speaking too softly. Project your voice as if speaking over a phone line. Use formal address ({formal_address}) when speaking to callers."
    
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
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
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
        
        # Start output processing thread immediately for greeting playback
        self.output_thread = threading.Thread(target=self._process_output_queue, daemon=True)
        self.output_thread.start()
        logger.info(f"🎵 Started output queue processing for session {session_id}")
        
        # Keep processing flag for backward compatibility with other parts of the code
        self.processing = self.output_processing
        
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
            
            # Record incoming audio
            if self.call_recorder:
                self.call_recorder.record_incoming_audio(pcm_data)
            
            # Queue audio for processing
            self.audio_queue.put(pcm_data)
            logger.info(f"🎤 Queued {len(pcm_data)} bytes of PCM audio for processing")
            
            # Start input processing if not already running
            if not self.input_processing:
                self.input_processing = True
                threading.Thread(target=self._process_audio_queue, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")
    
    def _process_audio_queue(self):
        """Process queued audio through voice session with improved async handling"""
        audio_buffer = b""
        chunk_size = 320  # 20ms at 8kHz for immediate processing
        
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
        """Async version of audio queue processing with proper streaming"""
        try:
            # Initialize voice session first
            if not await self.voice_session.initialize_voice_session():
                logger.error("Failed to initialize voice session")
                return
                
            # Start receive task for continuous response handling
            receive_task = asyncio.create_task(self._continuous_receive_responses())
            
            # Use smaller chunk size for lower latency
            # 20ms chunks for immediate response
            chunk_size = 320  # 20ms at 8kHz, 16-bit = 320 bytes
            min_chunk_size = 160  # 10ms minimum - send as soon as possible
            
            # No artificial timing delays - send immediately
            last_send_time = time.time()
            target_interval = 0  # No delay between chunks
            
            while self.input_processing:
                try:
                    # Get audio chunk with timeout - use asyncio-compatible approach
                    try:
                        # Use a small timeout to check if processing should continue
                        audio_chunk = self.audio_queue.get_nowait()
                        audio_buffer += audio_chunk
                    except queue.Empty:
                        # Process immediately if we have any audio
                        if len(audio_buffer) >= min_chunk_size and self.voice_session.gemini_session:
                            # Send whatever we have immediately - no padding needed
                            await self._send_audio_to_gemini(audio_buffer)
                            audio_buffer = b""
                        else:
                            # Only sleep if we have no audio at all
                            await asyncio.sleep(0.001)  # Minimal sleep just to yield
                        continue
                    
                    # Process audio immediately without timing delays
                    while len(audio_buffer) >= chunk_size:
                        chunk_to_process = audio_buffer[:chunk_size]
                        audio_buffer = audio_buffer[chunk_size:]
                        
                        # Send audio to Gemini immediately
                        if self.voice_session.gemini_session:
                            await self._send_audio_to_gemini(chunk_to_process)
                        
                except Exception as e:
                    logger.error(f"Error in async audio processing: {e}")
                    await asyncio.sleep(0.01)  # Minimal retry delay
                    
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
                                                    
                                                    # Send immediately for lowest latency
                                                    self.send_audio(telephony_audio)
                                                except Exception as e:
                                                    logger.error(f"Error decoding base64 audio: {e}")
                                            elif isinstance(audio_data, bytes):
                                                logger.info(f"📥 Received {len(audio_data)} bytes of audio from Gemini")
                                                # Convert and send in smaller chunks to avoid overwhelming the receiver
                                                telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_data)
                                                
                                                # Send immediately for lowest latency
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
                                # Send immediately for lowest latency
                                self.send_audio(telephony_audio)
                            elif isinstance(response.data, str):
                                try:
                                    audio_bytes = base64.b64decode(response.data)
                                    logger.info(f"📥 Received {len(audio_bytes)} bytes of audio from Gemini (direct base64)")
                                    telephony_audio = self.voice_session.convert_gemini_to_telephony(audio_bytes)
                                    # Send immediately for lowest latency
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
                if self.input_processing:  # Only log if we're still processing
                    logger.error(f"Error receiving from Gemini: {e}")
                    await asyncio.sleep(0.01)  # Minimal retry delay
                    
        logger.info("🎧 Stopped continuous response receiver")
    
    # Removed _process_chunk - we use continuous streaming instead
    
    def send_audio(self, pcm16: bytes):
        """
        Takes 16-bit mono PCM at 8000 Hz, μ-law encodes, then enqueues in 20 ms packets.
        """
        # Record outgoing audio before encoding
        if self.call_recorder:
            self.call_recorder.record_outgoing_audio(pcm16)
        
        ulaw = self.pcm_to_ulaw(pcm16)  # your existing encoder
        packet_size = 160  # 20 ms @ 8000 Hz, 1 byte/sample for G.711
        for i in range(0, len(ulaw), packet_size):
            self.output_queue.put(ulaw[i:i + packet_size])
    
    def play_greeting_file(self, greeting_file: str = "greeting.wav"):
        """Play a greeting WAV file at the start of the call"""
        try:
            if not os.path.exists(greeting_file):
                logger.warning(f"Greeting file {greeting_file} not found, skipping greeting")
                return
                
            logger.info(f"🎵 Playing greeting file: {greeting_file}")
            
            # Load WAV file
            with wave.open(greeting_file, 'rb') as wav_file:
                # Get WAV parameters
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                logger.info(f"Greeting file info: {frames} frames, {sample_rate}Hz, {channels}ch, {sample_width}bytes/sample")
                
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
                    
                    logger.info(f"Resampled from {sample_rate}Hz to 8000Hz")
                
                # Send the greeting audio
                logger.info(f"🎙️ Sending greeting audio: {len(audio_data)} bytes")
                self.send_audio(audio_data)
                
                logger.info("✅ Greeting played successfully")
                
        except Exception as e:
            logger.error(f"Error playing greeting file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue without greeting - don't let this break the call
    
    def _process_output_queue(self):
        """
        Dequeue G.711 payloads and transmit with RTP pacing.
        Emits silence only for packet loss concealment; no artificial noise.
        """
        ptime_ms = 20
        frame_bytes = 160  # 20 ms of G.711
        silence = b"\xff" * frame_bytes  # μ-law silence

        while self.output_processing:
            try:
                payload = self.output_queue.get(timeout=0.1)
            except Exception:
                # Idle: do not inject comfort noise; stay silent to avoid "radio" hiss
                continue

            # Transmit exactly one frame per ptime
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
                        
                        # Play greeting after a short delay to ensure RTP is ready
                        def play_greeting_delayed():
                            time.sleep(0.5)  # Small delay to ensure RTP connection is established
                            logger.info("🎵 Playing greeting to caller...")
                            rtp_session.play_greeting_file("greeting.wav")
                        
                        threading.Thread(target=play_greeting_delayed, daemon=True).start()
                        
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
                        def start_conversation():
                            time.sleep(0.5)  # Small delay
                            logger.info("🎤 Starting conversation with called party...")
                            # The AI will start talking based on its system prompt
                        
                        threading.Thread(target=start_conversation, daemon=True).start()
                        
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
        if 'Bulgarian' in system_text:
            detected_lang = 'Bulgarian'
        elif 'English' in system_text:
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
            if 'Bulgarian' in system_text:
                detected_lang = 'Bulgarian'
            elif 'English' in system_text:
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
                            
                            if audio_files:  # Only include if audio files exist
                                recordings.append({
                                    "session_id": session_info.get('session_id'),
                                    "caller_id": session_info.get('caller_id'),
                                    "called_number": session_info.get('called_number'),
                                    "start_time": session_info.get('start_time'),
                                    "end_time": session_info.get('end_time'),
                                    "duration_seconds": session_info.get('duration_seconds'),
                                    "audio_files": audio_files
                                })
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
    logger.info("  GET /api/recordings - List call recordings")
    logger.info("  POST /api/make_call - Initiate outbound call")
    logger.info("=" * 60)
    
    uvicorn.run(
        "windows_voice_agent:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
