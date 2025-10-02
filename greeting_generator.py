"""
Google AI Studio Greeting Generator using Playwright
Generates custom greeting audio files in different languages
"""

import asyncio
import os
import base64
import json
import logging
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import aiofiles
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GreetingGenerator:
    """Generates custom greeting audio files using Google AI Studio"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.context = None
        self.playwright = None
        self.sessions_dir = Path("sessions")
        self.greetings_dir = Path("greetings")
        self.greetings_dir.mkdir(exist_ok=True)
        self.download_dir = None
        
    async def initialize(self):
        """Initialize Playwright browser"""
        try:
            logger.info("🎭 Initializing Playwright browser...")
            self.playwright = await async_playwright().start()
            
            # Create temporary download directory
            self.download_dir = tempfile.mkdtemp()
            
            # Launch browser in non-headless mode for debugging
            # Change to headless=True for production
            self.browser = await self.playwright.chromium.launch(
                headless=False,  # Set to True for production
                args=['--disable-blink-features=AutomationControlled']
            )
            
            # Create browser context with download handling
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                accept_downloads=True
            )
            
            # Set up download handling
            self.page = await self.context.new_page()
            
            # Listen for download events
            self.page.on("download", self._handle_download)
            
            logger.info("✅ Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    async def close(self):
        """Close browser and cleanup"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        if self.download_dir and os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)
        logger.info("🎭 Browser closed")
    
    async def _handle_download(self, download):
        """Handle download events"""
        try:
            # Save the download
            download_path = os.path.join(self.download_dir, download.suggested_filename)
            await download.save_as(download_path)
            logger.info(f"📥 Downloaded file: {download_path}")
        except Exception as e:
            logger.error(f"Error handling download: {e}")
    
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
        Generate greeting audio file using Google AI Studio
        
        Returns:
            Tuple of (audio_file_path, transcript_text) or (None, None) on failure
        """
        try:
            if not self.page:
                await self.initialize()
            
            logger.info(f"🌐 Navigating to Google AI Studio...")
            
            # Navigate to AI Studio Text-to-Speech
            # Using the direct TTS demo page
            await self.page.goto('https://cloud.google.com/text-to-speech#demo', wait_until='networkidle')
            
            # Wait for page to load
            await asyncio.sleep(2)
            
            # Generate greeting text based on language
            greeting_text = self._generate_greeting_text(
                language, language_code, company_name, caller_name, product_name, custom_text
            )
            
            logger.info(f"📝 Generated greeting text: {greeting_text}")
            
            # Try to find the text input area
            # Look for textarea or input field
            text_input = None
            selectors = [
                'textarea[placeholder*="Enter text"]',
                'textarea[placeholder*="Type text"]',
                'textarea.text-input',
                'textarea#text-input',
                'textarea[name="text"]',
                'div[contenteditable="true"]',
                'input[type="text"]'
            ]
            
            for selector in selectors:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=5000)
                    if element:
                        text_input = element
                        logger.info(f"✅ Found text input with selector: {selector}")
                        break
                except:
                    continue
            
            if not text_input:
                logger.error("❌ Could not find text input field")
                # Try alternative approach - use Google Cloud TTS API demo
                return await self._try_cloud_tts_demo(greeting_text, language_code)
            
            # Clear and enter text
            await text_input.clear()
            await text_input.type(greeting_text)
            
            # Look for voice selection
            await self._select_voice(language_code)
            
            # Look for play/generate button
            play_button = None
            button_selectors = [
                'button:has-text("Speak it")',
                'button:has-text("Play")',
                'button:has-text("Generate")',
                'button:has-text("Listen")',
                'button[aria-label*="play"]',
                'button[aria-label*="speak"]'
            ]
            
            for selector in button_selectors:
                try:
                    button = await self.page.wait_for_selector(selector, timeout=3000)
                    if button:
                        play_button = button
                        logger.info(f"✅ Found play button with selector: {selector}")
                        break
                except:
                    continue
            
            if play_button:
                await play_button.click()
                logger.info("🎵 Generating audio...")
                await asyncio.sleep(3)  # Wait for generation
            
            # Try to capture audio
            audio_path = await self._capture_audio(greeting_text, language_code)
            
            if audio_path:
                return str(audio_path), greeting_text
            else:
                logger.warning("⚠️ Could not capture audio, trying fallback method")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ Error generating greeting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    async def _try_cloud_tts_demo(self, greeting_text: str, language_code: str) -> Tuple[Optional[str], Optional[str]]:
        """Try using Google Cloud TTS demo page as fallback"""
        try:
            logger.info("🔄 Trying Google Cloud TTS demo page...")
            
            # Navigate to the simpler demo
            await self.page.goto('https://cloud.google.com/text-to-speech', wait_until='networkidle')
            await asyncio.sleep(2)
            
            # Look for the demo iframe or embedded demo
            # This is a simplified approach - actual implementation would need proper selectors
            logger.warning("⚠️ Google Cloud TTS demo requires manual interaction")
            
            # For now, return None to fall back to simple TTS
            return None, None
            
        except Exception as e:
            logger.error(f"Error with Cloud TTS demo: {e}")
            return None, None
    
    async def _select_voice(self, language_code: str):
        """Try to select appropriate voice"""
        try:
            # Look for voice dropdown
            voice_selectors = [
                'select[name="voice"]',
                'div[role="combobox"]',
                'button:has-text("Voice")'
            ]
            
            for selector in voice_selectors:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.click()
                        # Try to select Wavenet or Neural voice
                        await asyncio.sleep(1)
                        
                        # Look for voice options
                        if language_code == 'en':
                            await self.page.click('text="Wavenet-C"', timeout=2000)
                        # Add more language-specific voices as needed
                        
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Could not select voice: {e}")
    
    async def _capture_audio(self, greeting_text: str, language_code: str) -> Optional[Path]:
        """Try to capture generated audio"""
        try:
            # Check if any file was downloaded
            download_files = os.listdir(self.download_dir)
            if download_files:
                # Get the most recent download
                latest_file = max(
                    [os.path.join(self.download_dir, f) for f in download_files],
                    key=os.path.getctime
                )
                
                # Move to greetings directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"greeting_{language_code}_{timestamp}.wav"
                destination = self.greetings_dir / filename
                
                shutil.move(latest_file, destination)
                logger.info(f"✅ Saved audio file: {destination}")
                return destination
            
            # If no download, try to capture from audio element
            # This is a placeholder - actual implementation would need to record audio
            logger.warning("⚠️ No audio file downloaded")
            return None
            
        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            return None
    
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

# Singleton instance
greeting_generator = GreetingGenerator()

async def generate_greeting_for_lead(
    language: str,
    language_code: str,
    call_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate greeting for a specific lead
    
    Args:
        language: Full language name (e.g., "Bulgarian")
        language_code: Language code (e.g., "bg")
        call_config: Call configuration with company details
        
    Returns:
        Dict with greeting_file path and transcript
    """
    try:
        logger.warning("⚠️ Google AI Studio scraping is experimental and may require manual setup")
        logger.info("💡 For better results, using the simple TTS generator is recommended")
        
        # For now, return unsuccessful to fall back to simple TTS
        # The actual Google AI Studio integration requires:
        # 1. Authentication handling
        # 2. Complex page navigation
        # 3. Audio capture mechanisms
        
        return {
            "success": False,
            "error": "Google AI Studio scraping not fully implemented - use simple TTS instead"
        }
        
        # Original code kept for reference:
        # Initialize generator if needed
        # if not greeting_generator.page:
        #     await greeting_generator.initialize()
        
        # Extract config
        # company_name = call_config.get('company_name', 'Devoptics')
        # caller_name = call_config.get('caller_name', 'Assistant')
        # product_name = call_config.get('product_name', 'our product')
        
        # Generate greeting
        # audio_path, transcript = await greeting_generator.generate_greeting(
        #     language=language,
        #     language_code=language_code,
        #     company_name=company_name,
        #     caller_name=caller_name,
        #     product_name=product_name
        # )
        
        # if audio_path:
        #     return {
        #         "success": True,
        #         "greeting_file": audio_path,
        #         "transcript": transcript,
        #         "language": language,
        #         "language_code": language_code
        #     }
        # else:
        #     return {
        #         "success": False,
        #         "error": "Failed to generate greeting audio"
        #     }
            
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Test function
async def test_generator():
    """Test the greeting generator"""
    try:
        # This will now return unsuccessful and fall back to simple TTS
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
            logger.error(f"❌ Test failed - expected behavior: {result.get('error')}")
            logger.info("💡 The system will automatically fall back to simple TTS")
            
    finally:
        if greeting_generator.browser:
            await greeting_generator.close()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_generator())