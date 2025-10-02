"""
Test script for the greeting generator
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_greeting_generator():
    """Test the greeting generator with different languages"""
    
    try:
        # Try to import the greeting generator (priority order)
        try:
            from greeting_generator_hq import generate_greeting_for_lead
            logger.info("✅ Using HIGH-QUALITY greeting generator (edge-tts)")
            logger.info("🎤 Microsoft Neural voices - crystal clear audio!")
        except ImportError:
            try:
                from greeting_generator_simple import generate_greeting_for_lead
                logger.info("✅ Using simple greeting generator (gTTS)")
                logger.warning("⚠️ Audio may have some noise - install edge-tts for better quality")
            except ImportError:
                from greeting_generator import generate_greeting_for_lead
                logger.info("✅ Using advanced greeting generator (Playwright)")
    except ImportError as e:
        logger.error(f"❌ No greeting generator available: {e}")
        logger.error("💡 Run: setup_greeting_generator.bat")
        logger.error("💡 Or install manually: pip install edge-tts pydub")
        return
    
    # Test cases for different languages
    test_cases = [
        {
            "phone": "+359888123456",
            "expected_lang": "Bulgarian",
            "config": {
                "company_name": "QuantumAI",
                "caller_name": "Maria",
                "product_name": "АртроФлекс"
            }
        },
        {
            "phone": "+40721123456", 
            "expected_lang": "Romanian",
            "config": {
                "company_name": "TechCorp",
                "caller_name": "Ion",
                "product_name": "Software Suite"
            }
        },
        {
            "phone": "+49151123456",
            "expected_lang": "German", 
            "config": {
                "company_name": "Technik GmbH",
                "caller_name": "Hans",
                "product_name": "Premium Software"
            }
        },
        {
            "phone": "+1234567890",
            "expected_lang": "English",
            "config": {
                "company_name": "Devoptics",
                "caller_name": "John",
                "product_name": "FTD"
            }
        }
    ]
    
    logger.info("🧪 Starting greeting generator tests...")
    logger.info("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n📞 Test {i}: {test['expected_lang']} greeting")
        logger.info(f"   Phone: {test['phone']}")
        logger.info(f"   Company: {test['config']['company_name']}")
        logger.info(f"   Caller: {test['config']['caller_name']}")
        logger.info(f"   Product: {test['config']['product_name']}")
        
        # Detect language from phone number (simplified)
        if test['phone'].startswith('+359'):
            lang_code = 'bg'
        elif test['phone'].startswith('+40'):
            lang_code = 'ro'
        elif test['phone'].startswith('+49'):
            lang_code = 'de'
        else:
            lang_code = 'en'
        
        # Generate greeting
        try:
            result = await generate_greeting_for_lead(
                language=test['expected_lang'],
                language_code=lang_code,
                call_config=test['config']
            )
            
            if result.get("success"):
                logger.info(f"   ✅ Success! Audio file: {result['greeting_file']}")
                logger.info(f"   📝 Transcript: {result['transcript']}")
            else:
                logger.error(f"   ❌ Failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
        
        logger.info("-" * 60)
    
    logger.info("\n🎉 Test completed!")
    logger.info("Check the 'greetings' folder for generated audio files.")

if __name__ == "__main__":
    asyncio.run(test_greeting_generator())
