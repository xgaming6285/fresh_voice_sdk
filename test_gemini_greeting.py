"""
Test script for Gemini greeting generator (uses same Puck voice as calls)
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gemini_greeting():
    """Test the Gemini greeting generator"""
    
    try:
        from greeting_generator_gemini import generate_greeting_for_lead
        logger.info("✅ Gemini greeting generator imported successfully")
        logger.info("🎤 This will use the same Puck voice as your voice calls")
    except ImportError as e:
        logger.error(f"❌ Failed to import Gemini greeting generator: {e}")
        logger.error("💡 Make sure google-genai is installed: pip install google-genai")
        return
    
    # Test Bulgarian greeting
    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing Bulgarian greeting with Puck voice")
    logger.info("=" * 60)
    
    result = await generate_greeting_for_lead(
        language="Bulgarian",
        language_code="bg",
        call_config={
            "company_name": "QuantumAI",
            "caller_name": "Maria",
            "product_name": "АртроФлекс"
        }
    )
    
    if result.get("success"):
        logger.info("\n✅ SUCCESS! Greeting generated with Puck voice")
        logger.info(f"📁 Audio file: {result['greeting_file']}")
        logger.info(f"📝 Transcript: {result['transcript']}")
        logger.info(f"🗣️ Language: {result['language']}")
        logger.info(f"🎤 Method: {result['method']}")
        logger.info("\n💡 This is the SAME voice used in your voice calls!")
        logger.info("💡 Play the audio file to verify quality")
    else:
        logger.error(f"\n❌ FAILED: {result.get('error')}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Make sure GOOGLE_API_KEY environment variable is set")
        logger.error("2. Check that you have google-genai installed: pip install google-genai")
        logger.error("3. Verify your API key has access to Gemini Live API")

if __name__ == "__main__":
    asyncio.run(test_gemini_greeting())
