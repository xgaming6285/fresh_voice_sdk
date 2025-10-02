"""
High-Quality Greeting Generator using edge-tts (Microsoft Edge TTS)
Produces clean audio without radio noise
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

# Try to import edge-tts (high quality Microsoft TTS)
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    logger.info("✅ edge-tts library loaded for high-quality TTS")
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("⚠️ edge-tts not available - install with: pip install edge-tts")

class HighQualityGreetingGenerator:
    """Generates high-quality greeting audio files using Microsoft Edge TTS"""
    
    def __init__(self):
        self.greetings_dir = Path("greetings")
        self.greetings_dir.mkdir(exist_ok=True)
        
        # Voice mapping for different languages (Microsoft Edge voices)
        self.voice_map = {
            'bg': 'bg-BG-BorislavNeural',     # Bulgarian male
            'ro': 'ro-RO-EmilNeural',          # Romanian male
            'el': 'el-GR-NestorasNeural',      # Greek male
            'de': 'de-DE-ConradNeural',        # German male
            'fr': 'fr-FR-HenriNeural',         # French male
            'es': 'es-ES-AlvaroNeural',        # Spanish male
            'it': 'it-IT-DiegoNeural',         # Italian male
            'ru': 'ru-RU-DmitryNeural',        # Russian male
            'en': 'en-US-ChristopherNeural',   # English male
            'pl': 'pl-PL-MarekNeural',         # Polish male
            'cs': 'cs-CZ-AntoninNeural',       # Czech male
            'sk': 'sk-SK-LukasNeural',         # Slovak male
            'hu': 'hu-HU-NoemiNeural',         # Hungarian female
            'hr': 'hr-HR-SreckoNeural',        # Croatian male
            'uk': 'uk-UA-OstapNeural',         # Ukrainian male
            'tr': 'tr-TR-AhmetNeural',         # Turkish male
            'pt': 'pt-PT-DuarteNeural',        # Portuguese male
            'nl': 'nl-NL-MaartenNeural',       # Dutch male
            'sv': 'sv-SE-MattiasNeural',       # Swedish male
            'da': 'da-DK-JeppeNeural',         # Danish male
            'no': 'nb-NO-FinnNeural',          # Norwegian male
            'fi': 'fi-FI-HarriNeural',         # Finnish male
            'ar': 'ar-SA-HamedNeural',         # Arabic male
            'he': 'he-IL-AvriNeural',          # Hebrew male
            'ja': 'ja-JP-KeitaNeural',         # Japanese male
            'ko': 'ko-KR-InJoonNeural',        # Korean male
            'zh': 'zh-CN-YunxiNeural',         # Chinese male
            'hi': 'hi-IN-MadhurNeural',        # Hindi male
        }
        
        # Female voice alternatives (for variety)
        self.female_voice_map = {
            'bg': 'bg-BG-KalinaNeural',        # Bulgarian female
            'ro': 'ro-RO-AlinaNeural',         # Romanian female
            'el': 'el-GR-AthinaNeural',        # Greek female
            'de': 'de-DE-KatjaNeural',         # German female
            'fr': 'fr-FR-DeniseNeural',        # French female
            'es': 'es-ES-ElviraNeural',        # Spanish female
            'it': 'it-IT-ElsaNeural',          # Italian female
            'ru': 'ru-RU-SvetlanaNeural',      # Russian female
            'en': 'en-US-JennyNeural',         # English female
        }
    
    async def generate_greeting(
        self, 
        language: str, 
        language_code: str,
        company_name: str = "Devoptics",
        caller_name: str = "John",
        product_name: str = "FTD",
        custom_text: Optional[str] = None,
        use_female_voice: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate high-quality greeting audio file using Microsoft Edge TTS
        
        Returns:
            Tuple of (audio_file_path, transcript_text) or (None, None) on failure
        """
        try:
            if not EDGE_TTS_AVAILABLE:
                logger.error("❌ edge-tts not available")
                return None, None
            
            # Generate greeting text based on language
            greeting_text = self._generate_greeting_text(
                language, language_code, company_name, caller_name, product_name, custom_text
            )
            
            logger.info(f"🌐 Generating high-quality greeting for {language}")
            logger.info(f"📝 Text: {greeting_text}")
            
            # Select appropriate voice
            voice_name = self._get_voice_name(language_code, use_female_voice)
            logger.info(f"🎤 Using voice: {voice_name}")
            
            # Create TTS communication object
            communicate = edge_tts.Communicate(greeting_text, voice_name)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"greeting_{language_code}_{timestamp}.mp3"
            filepath = self.greetings_dir / filename
            
            # Generate audio
            await communicate.save(str(filepath))
            
            # Convert to WAV for better compatibility
            wav_filepath = await self._convert_to_wav(filepath)
            
            logger.info(f"✅ Saved high-quality greeting audio: {wav_filepath}")
            
            return str(wav_filepath), greeting_text
            
        except Exception as e:
            logger.error(f"❌ Error generating greeting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _get_voice_name(self, language_code: str, use_female_voice: bool = False) -> str:
        """Get the appropriate voice name for the language"""
        if use_female_voice and language_code in self.female_voice_map:
            return self.female_voice_map[language_code]
        return self.voice_map.get(language_code, 'en-US-ChristopherNeural')
    
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
        
        # Language-specific greetings with proper pronunciation hints
        greetings = {
            'bg': f"Здравейте! Аз съм {caller_name} от компания {company_name}. Обаждам се във връзка с {product_name}. Интересувате ли се да научите повече?",
            'ro': f"Bună ziua! Sunt {caller_name} de la compania {company_name}. Vă sun în legătură cu {product_name}. Sunteți interesat să aflați mai multe?",
            'el': f"Γεια σας! Είμαι ο {caller_name} από την εταιρεία {company_name}. Σας καλώ σχετικά με το {product_name}. Θα θέλατε να μάθετε περισσότερα;",
            'de': f"Guten Tag! Ich bin {caller_name} von {company_name}. Ich rufe Sie bezüglich {product_name} an. Hätten Sie Interesse, mehr zu erfahren?",
            'fr': f"Bonjour! Je suis {caller_name} de la société {company_name}. Je vous appelle au sujet de {product_name}. Seriez-vous intéressé pour en savoir plus?",
            'es': f"¡Buenos días! Soy {caller_name} de la empresa {company_name}. Le llamo para hablarle sobre {product_name}. ¿Le gustaría saber más?",
            'it': f"Buongiorno! Sono {caller_name} dell'azienda {company_name}. La chiamo per parlarle di {product_name}. Le interesserebbe saperne di più?",
            'ru': f"Здравствуйте! Меня зовут {caller_name}, я представляю компанию {company_name}. Звоню вам по поводу {product_name}. Вам было бы интересно узнать больше?",
            'en': f"Hello! I'm {caller_name} from {company_name}. I'm calling about {product_name}. Would you be interested in learning more?",
        }
        
        # Get greeting or default to English
        return greetings.get(language_code, greetings['en'])
    
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
                
                # Apply some audio processing to improve quality
                # Normalize audio
                audio = audio.normalize()
                
                # Apply compression to even out volume
                audio = audio.compress_dynamic_range()
                
                # Save as WAV
                wav_path = mp3_path.with_suffix('.wav')
                audio.export(str(wav_path), format="wav")
                
                # Remove MP3
                mp3_path.unlink()
                
                logger.info("✅ Converted MP3 to high-quality WAV")
                return wav_path
                
            except ImportError:
                logger.warning("⚠️ pydub not available, keeping MP3 format")
                return mp3_path
                
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            return mp3_path
    
    async def list_available_voices(self) -> Dict[str, list]:
        """List all available voices for each language"""
        try:
            if not EDGE_TTS_AVAILABLE:
                return {}
            
            voices = await edge_tts.list_voices()
            
            # Group by language
            voices_by_language = {}
            for voice in voices:
                lang = voice['Locale'][:2]  # Get language code
                if lang not in voices_by_language:
                    voices_by_language[lang] = []
                voices_by_language[lang].append({
                    'name': voice['ShortName'],
                    'gender': voice['Gender'],
                    'locale': voice['Locale']
                })
            
            return voices_by_language
            
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return {}

# Singleton instance
hq_greeting_generator = HighQualityGreetingGenerator()

async def generate_greeting_for_lead(
    language: str,
    language_code: str,
    call_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate high-quality greeting for a specific lead
    
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
        
        # Determine if we should use female voice based on caller name
        use_female = any(name in caller_name.lower() for name in ['maria', 'sarah', 'anna', 'elena', 'sofia'])
        
        # Generate greeting
        audio_path, transcript = await hq_greeting_generator.generate_greeting(
            language=language,
            language_code=language_code,
            company_name=company_name,
            caller_name=caller_name,
            product_name=product_name,
            use_female_voice=use_female
        )
        
        if audio_path:
            return {
                "success": True,
                "greeting_file": audio_path,
                "transcript": transcript,
                "language": language,
                "language_code": language_code,
                "method": "edge-tts (high-quality)"
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
    """Test the high-quality greeting generator"""
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
            logger.info(f"🎤 Method: {result['method']}")
        else:
            logger.error(f"❌ Test failed: {result.get('error')}")
        
        # List available voices
        logger.info("\n📋 Available voices by language:")
        voices = await hq_greeting_generator.list_available_voices()
        for lang, voice_list in list(voices.items())[:5]:  # Show first 5 languages
            logger.info(f"{lang}: {len(voice_list)} voices available")
            
    except Exception as e:
        logger.error(f"Test error: {e}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_generator())
