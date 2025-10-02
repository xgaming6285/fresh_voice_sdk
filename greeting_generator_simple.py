"""
Simple Greeting Generator using gTTS (Google Translate TTS)
This is a fallback solution that doesn't require Google AI Studio authentication
"""

import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("✅ gTTS library loaded for simple TTS")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("⚠️ gTTS not available - install with: pip install gTTS")

class SimpleGreetingGenerator:
    """Generates greeting audio files using Google Translate TTS"""
    
    def __init__(self):
        self.greetings_dir = Path("greetings")
        self.greetings_dir.mkdir(exist_ok=True)
    
    async def generate_greeting(
        self, 
        language: str, 
        language_code: str,
        company_name: str = "Devoptics",
        caller_name: str = "John",
        product_name: str = "FTD",
        custom_text: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate greeting audio file using gTTS
        
        Returns:
            Tuple of (audio_file_path, transcript_text) or (None, None) on failure
        """
        try:
            if not GTTS_AVAILABLE:
                logger.error("❌ gTTS not available")
                return None, None
            
            # Generate greeting text based on language
            greeting_text = self._generate_greeting_text(
                language, language_code, company_name, caller_name, product_name, custom_text
            )
            
            logger.info(f"🌐 Generating greeting for {language}")
            logger.info(f"📝 Text: {greeting_text}")
            
            # Map language codes to gTTS language codes
            gtts_lang = self._map_language_code(language_code)
            
            # Generate audio using gTTS
            tts = gTTS(text=greeting_text, lang=gtts_lang, slow=False)
            
            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"greeting_{language_code}_{timestamp}.mp3"
            filepath = self.greetings_dir / filename
            
            # Save as MP3 (gTTS default)
            tts.save(str(filepath))
            
            # Convert to WAV if needed (optional)
            wav_filepath = await self._convert_to_wav(filepath)
            
            logger.info(f"✅ Saved greeting audio: {wav_filepath}")
            
            return str(wav_filepath), greeting_text
            
        except Exception as e:
            logger.error(f"❌ Error generating greeting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _generate_greeting_text(
        self,
        language: str,
        language_code: str,
        company_name: str,
        caller_name: str,
        product_name: str,
        custom_text: Optional[str]
    ) -> str:
        """Generate greeting text in the appropriate language"""
        
        if custom_text:
            return custom_text
        
        # Language-specific greetings
        greetings = {
            'bg': f"Здравейте, аз съм {caller_name} от {company_name}. Интересувате ли се от {product_name}?",
            'ro': f"Bună ziua, sunt {caller_name} de la {company_name}. Sunteți interesat de {product_name}?",
            'el': f"Γεια σας, είμαι ο {caller_name} από την {company_name}. Ενδιαφέρεστε για το {product_name};",
            'de': f"Hallo, ich bin {caller_name} von {company_name}. Sind Sie an {product_name} interessiert?",
            'fr': f"Bonjour, je suis {caller_name} de {company_name}. Êtes-vous intéressé par {product_name}?",
            'es': f"Hola, soy {caller_name} de {company_name}. ¿Está interesado en {product_name}?",
            'it': f"Buongiorno, sono {caller_name} di {company_name}. È interessato a {product_name}?",
            'ru': f"Здравствуйте, я {caller_name} из {company_name}. Вас интересует {product_name}?",
            'en': f"Hello, I'm {caller_name} from {company_name}. Are you interested in {product_name}?",
        }
        
        # Get greeting or default to English
        return greetings.get(language_code, greetings['en'])
    
    def _map_language_code(self, language_code: str) -> str:
        """Map our language codes to gTTS language codes"""
        mapping = {
            'bg': 'bg',     # Bulgarian
            'ro': 'ro',     # Romanian
            'el': 'el',     # Greek
            'de': 'de',     # German
            'fr': 'fr',     # French
            'es': 'es',     # Spanish
            'it': 'it',     # Italian
            'ru': 'ru',     # Russian
            'en': 'en',     # English
            'zh': 'zh',     # Chinese
            'ja': 'ja',     # Japanese
            'ko': 'ko',     # Korean
            'ar': 'ar',     # Arabic
            'pt': 'pt',     # Portuguese
            'pt-BR': 'pt', # Portuguese (Brazil)
            'nl': 'nl',     # Dutch
            'pl': 'pl',     # Polish
            'tr': 'tr',     # Turkish
            'sv': 'sv',     # Swedish
            'no': 'no',     # Norwegian
            'da': 'da',     # Danish
            'fi': 'fi',     # Finnish
            'cs': 'cs',     # Czech
            'sk': 'sk',     # Slovak
            'hu': 'hu',     # Hungarian
            'hr': 'hr',     # Croatian
            'uk': 'uk',     # Ukrainian
        }
        
        return mapping.get(language_code, 'en')
    
    async def _convert_to_wav(self, mp3_path: Path) -> Path:
        """Convert MP3 to WAV format using available tools"""
        try:
            # Try using pydub if available
            try:
                from pydub import AudioSegment
                
                # Load MP3
                audio = AudioSegment.from_mp3(str(mp3_path))
                
                # Convert to 8kHz mono WAV (telephony format)
                audio = audio.set_frame_rate(8000)
                audio = audio.set_channels(1)
                audio = audio.set_sample_width(2)  # 16-bit
                
                # Save as WAV
                wav_path = mp3_path.with_suffix('.wav')
                audio.export(str(wav_path), format="wav")
                
                # Remove MP3
                mp3_path.unlink()
                
                logger.info("✅ Converted MP3 to WAV using pydub")
                return wav_path
                
            except ImportError:
                logger.warning("⚠️ pydub not available, keeping MP3 format")
                # Note: The voice agent may need to handle MP3 format
                return mp3_path
                
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            return mp3_path

# Singleton instance
simple_greeting_generator = SimpleGreetingGenerator()

async def generate_greeting_for_lead(
    language: str,
    language_code: str,
    call_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate greeting for a specific lead using simple TTS
    
    Args:
        language: Full language name (e.g., "Bulgarian")
        language_code: Language code (e.g., "bg")
        call_config: Call configuration with company details
        
    Returns:
        Dict with greeting_file path and transcript
    """
    try:
        # Extract config
        company_name = call_config.get('company_name', 'Devoptics')
        caller_name = call_config.get('caller_name', 'Assistant')
        product_name = call_config.get('product_name', 'our product')
        
        # Generate greeting
        audio_path, transcript = await simple_greeting_generator.generate_greeting(
            language=language,
            language_code=language_code,
            company_name=company_name,
            caller_name=caller_name,
            product_name=product_name
        )
        
        if audio_path:
            return {
                "success": True,
                "greeting_file": audio_path,
                "transcript": transcript,
                "language": language,
                "language_code": language_code,
                "method": "gTTS"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate greeting audio"
            }
            
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Test function
async def test_generator():
    """Test the simple greeting generator"""
    try:
        # Test Bulgarian greeting
        result = await generate_greeting_for_lead(
            language="Bulgarian",
            language_code="bg",
            call_config={
                "company_name": "QuantumAI",
                "caller_name": "Maria",
                "product_name": "АртроФлекс"
            }
        )
        
        if result["success"]:
            logger.info(f"✅ Test successful! Audio saved to: {result['greeting_file']}")
            logger.info(f"📝 Transcript: {result['transcript']}")
        else:
            logger.error(f"❌ Test failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Test error: {e}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_generator())
