"""
Gemini Live API Greeting Generator
Uses the same Puck voice as the voice calls for consistent audio quality
No web scraping needed - direct API integration
"""

import asyncio
import os
import base64
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.signal import resample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Gemini client
try:
    from google import genai
    GENAI_AVAILABLE = True
    logger.info("✅ Google GenAI library loaded")
except ImportError:
    GENAI_AVAILABLE = False
    logger.error("❌ google-genai not available - install with: pip install google-genai")

class GeminiGreetingGenerator:
    """Generates greeting audio files using Gemini Live API (same as voice calls)"""
    
    def __init__(self):
        self.greetings_dir = Path("greetings")
        self.greetings_dir.mkdir(exist_ok=True)
        self.client = None
        
        if GENAI_AVAILABLE:
            try:
                # Get API key from environment
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    logger.error("❌ GOOGLE_API_KEY environment variable not set")
                    logger.error("💡 Set it with: set GOOGLE_API_KEY=your-key-here (Windows)")
                    self.client = None
                    return
                
                # Initialize Gemini client with API key
                self.client = genai.Client(
                    api_key=api_key,
                    http_options={"api_version": "v1beta"}
                )
                logger.info("✅ Gemini client initialized successfully")
                logger.info(f"🔑 Using API key: {api_key[:20]}...{api_key[-4:]}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini client: {e}")
                self.client = None
    
    async def generate_greeting(
        self, 
        language: str, 
        language_code: str,
        company_name: str = "Devoptics",
        caller_name: str = "John",
        product_name: str = "FTD",
        custom_text: Optional[str] = None,
        voice_name: str = "Puck",
        greeting_instruction: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate greeting audio file using Gemini Live API with Puck voice
        
        Returns:
            Tuple of (audio_file_path, transcript_text) or (None, None) on failure
        """
        try:
            if not GENAI_AVAILABLE or not self.client:
                logger.error("❌ Gemini client not available")
                return None, None
            
            # Generate greeting text based on language
            greeting_text = self._generate_greeting_text(
                language, language_code, company_name, caller_name, product_name, custom_text, greeting_instruction
            )
            
            logger.info(f"🌐 Generating greeting with Gemini Live API ({voice_name} voice)")
            logger.info(f"🗣️ Language: {language}")
            logger.info(f"📝 Text: {greeting_text}")
            
            # Create voice config (same as used in calls)
            voice_config = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice_name
                        }
                    }
                },
                "system_instruction": {
                    "parts": [
                        {
                            "text": f"You are a professional voice actor. Say exactly this in {language}: \"{greeting_text}\". Speak clearly and naturally."
                        }
                    ]
                }
            }
            
            # Create Gemini Live session
            logger.info("🎤 Starting Gemini Live session...")
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-live-001",
                config=voice_config
            ) as session:
                
                # Send a simple message to trigger the greeting
                logger.info("📤 Requesting audio generation...")
                await session.send(
                    input="Please say the greeting now.",
                    end_of_turn=True
                )
                
                # Collect audio responses
                audio_chunks = []
                turn_complete = False
                last_chunk_time = asyncio.get_event_loop().time()
                
                logger.info("📥 Receiving audio from Gemini...")
                async for response in session.receive():
                    # Extract audio data from response
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if hasattr(part.inline_data, 'mime_type') and 'audio' in part.inline_data.mime_type:
                                        audio_data = part.inline_data.data
                                        
                                        # Handle both string (base64) and bytes
                                        if isinstance(audio_data, str):
                                            audio_bytes = base64.b64decode(audio_data)
                                        else:
                                            audio_bytes = audio_data
                                        
                                        if len(audio_bytes) > 0:
                                            audio_chunks.append(audio_bytes)
                                            logger.info(f"📥 Received chunk {len(audio_chunks)}: {len(audio_bytes)} bytes")
                                            last_chunk_time = asyncio.get_event_loop().time()
                                
                                # Also log text responses
                                if hasattr(part, 'text') and part.text:
                                    logger.debug(f"Text: {part.text}")
                        
                        # Check if turn is complete
                        if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                            logger.info("✅ Turn complete - received all audio")
                            turn_complete = True
                            # Wait a tiny bit more for any final chunks
                            await asyncio.sleep(0.1)
                            break
                    
                    # Stop if we haven't received audio for 2 seconds (stream ended)
                    current_time = asyncio.get_event_loop().time()
                    if len(audio_chunks) > 0 and (current_time - last_chunk_time) > 2.0:
                        logger.info(f"📊 No more audio for 2s, stopping with {len(audio_chunks)} chunks")
                        break
                
                if not audio_chunks:
                    logger.error("❌ No audio received from Gemini")
                    return None, None
                
                # Combine audio chunks
                logger.info(f"🔊 Combining {len(audio_chunks)} audio chunks...")
                combined_audio = b''.join(audio_chunks)
                
                # Save audio file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"greeting_{language_code}_{timestamp}.wav"
                filepath = self.greetings_dir / filename
                
                # Convert from Gemini format (24kHz) to telephony format (8kHz)
                telephony_audio = self._convert_to_telephony_format(combined_audio)
                
                # Save as WAV
                self._save_as_wav(telephony_audio, filepath)
                
                logger.info(f"✅ Saved greeting audio: {filepath}")
                logger.info(f"🎤 Voice: Puck (same as voice calls)")
                
                return str(filepath), greeting_text
                
        except Exception as e:
            logger.error(f"❌ Error generating greeting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _convert_to_telephony_format(self, gemini_audio: bytes) -> bytes:
        """
        Convert Gemini audio (24kHz) to telephony format (8kHz)
        Same conversion used in the voice agent
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(gemini_audio, dtype=np.int16)
            
            # Resample from 24kHz to 8kHz
            num_samples_in = len(audio_array)
            num_samples_out = int(num_samples_in * 8000 / 24000)
            
            resampled_array = resample(audio_array, num_samples_out)
            
            # Convert back to int16
            telephony_audio = resampled_array.astype(np.int16).tobytes()
            
            logger.info(f"🔄 Converted audio: {len(gemini_audio)} bytes (24kHz) → {len(telephony_audio)} bytes (8kHz)")
            
            return telephony_audio
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return gemini_audio
    
    def _save_as_wav(self, audio_data: bytes, filepath: Path):
        """Save audio data as WAV file"""
        try:
            import wave
            
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)   # 16-bit
                wav_file.setframerate(8000)  # 8kHz
                wav_file.writeframes(audio_data)
            
            logger.info(f"💾 Saved as WAV: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}")
            raise
    
    def _generate_greeting_text(
        self,
        language: str,
        language_code: str,
        company_name: str,
        caller_name: str,
        product_name: str,
        custom_text: Optional[str],
        greeting_instruction: Optional[str] = None
    ) -> str:
        """Generate greeting text in the appropriate language"""
        
        # Priority: greeting_instruction > custom_text > default templates
        if greeting_instruction:
            # User provided specific instruction - use it directly
            return greeting_instruction
        
        if custom_text:
            return custom_text
        
        # Language-specific greetings
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

# Singleton instance
gemini_greeting_generator = GeminiGreetingGenerator()

async def generate_greeting_for_lead(
    language: str,
    language_code: str,
    call_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate greeting for a specific lead using Gemini Live API
    
    Args:
        language: Full language name (e.g., "Bulgarian")
        language_code: Language code (e.g., "bg")
        call_config: Call configuration with company details, voice, and greeting instruction
        
    Returns:
        Dict with greeting_file path and transcript
    """
    try:
        # Extract config
        company_name = call_config.get('company_name', 'Devoptics')
        caller_name = call_config.get('caller_name', 'Assistant')
        product_name = call_config.get('product_name', 'our product')
        voice_name = call_config.get('voice_name', 'Puck')
        greeting_instruction = call_config.get('greeting_instruction')
        
        # Generate greeting
        audio_path, transcript = await gemini_greeting_generator.generate_greeting(
            language=language,
            language_code=language_code,
            company_name=company_name,
            caller_name=caller_name,
            product_name=product_name,
            voice_name=voice_name,
            greeting_instruction=greeting_instruction
        )
        
        if audio_path:
            return {
                "success": True,
                "greeting_file": audio_path,
                "transcript": transcript,
                "language": language,
                "language_code": language_code,
                "voice": voice_name,
                "method": f"gemini-live ({voice_name} voice)"
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
    """Test the Gemini greeting generator"""
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
            
    except Exception as e:
        logger.error(f"Test error: {e}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_generator())
