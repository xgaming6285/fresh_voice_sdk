"""
Quick test script for edge-tts high-quality audio
"""

import asyncio
import edge_tts

async def test_edge_tts():
    """Test edge-tts with a simple greeting"""
    
    # Test text in different languages
    tests = [
        ("en-US-ChristopherNeural", "Hello! I'm calling from QuantumAI about our amazing product."),
        ("bg-BG-BorislavNeural", "Здравейте! Обаждам се от QuantumAI за нашия невероятен продукт."),
        ("de-DE-ConradNeural", "Guten Tag! Ich rufe von QuantumAI an wegen unserem tollen Produkt."),
    ]
    
    print("🎤 Testing edge-tts high-quality voices...")
    print("=" * 60)
    
    for voice, text in tests:
        print(f"\n📢 Voice: {voice}")
        print(f"📝 Text: {text}")
        
        # Create communication object
        communicate = edge_tts.Communicate(text, voice)
        
        # Generate filename
        filename = f"test_{voice.split('-')[0]}.mp3"
        
        # Save audio
        await communicate.save(filename)
        print(f"✅ Saved: {filename}")
    
    print("\n🎉 Test complete! Check the generated MP3 files.")
    print("💡 Notice: No radio noise, crystal-clear audio!")

if __name__ == "__main__":
    asyncio.run(test_edge_tts())
