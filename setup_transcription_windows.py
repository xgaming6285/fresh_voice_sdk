#!/usr/bin/env python3
"""
Windows Transcription Setup Script

This script helps install the best transcription solution for Windows.
It tries to install faster-whisper first (most Windows-compatible), then offers alternatives.

Usage: python setup_transcription_windows.py
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_package(package, description):
    """Try to install a package"""
    print(f"\n🔄 Installing {description}...")
    print(f"   Command: pip install {package}")
    
    success, stdout, stderr = run_command(f"pip install {package}")
    
    if success:
        print(f"✅ Successfully installed {description}!")
        return True
    else:
        print(f"❌ Failed to install {description}")
        print(f"   Error: {stderr}")
        return False

def test_import(module_name, package_name):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {package_name} is working!")
        return True
    except ImportError as e:
        print(f"❌ {package_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {package_name} import error: {e}")
        return False

def main():
    print("=" * 60)
    print("🎤 Windows Transcription Setup Script")
    print("=" * 60)
    print("This script will install the best transcription solution for Windows.")
    print("We'll try faster-whisper first (most compatible), then alternatives.")
    print()
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Option 1: Try faster-whisper (most Windows-compatible)
    print("\n" + "="*40)
    print("Option 1: faster-whisper (Recommended)")
    print("="*40)
    print("✅ No numba/llvmlite dependencies")
    print("✅ Faster than original whisper")
    print("✅ Windows-friendly")
    
    if install_package("faster-whisper>=0.10.0", "faster-whisper"):
        if test_import("faster_whisper", "faster-whisper"):
            print("\n🎉 SUCCESS! faster-whisper is ready to use!")
            print("\n📝 Your voice agent will now use faster-whisper for transcription.")
            return
    
    # Option 2: OpenAI API
    print("\n" + "="*40)
    print("Option 2: OpenAI API (Cloud-based)")
    print("="*40)
    print("✅ Always compatible")
    print("⚠️ Requires internet connection")
    print("⚠️ Requires OpenAI API key")
    
    choice = input("\nDo you want to try OpenAI API instead? (y/n): ").lower()
    if choice in ['y', 'yes']:
        if install_package("openai>=1.0.0", "OpenAI Python client"):
            if test_import("openai", "OpenAI client"):
                print("\n✅ OpenAI client installed successfully!")
                print("\n📝 To use OpenAI API transcription:")
                print("   1. Get an API key from: https://platform.openai.com/api-keys")
                print("   2. Set environment variable: set OPENAI_API_KEY=your_key_here")
                print("   3. Or create .env file with: OPENAI_API_KEY=your_key_here")
                return
    
    # Option 3: Original whisper (last resort)
    print("\n" + "="*40)
    print("Option 3: Original openai-whisper (Last resort)")
    print("="*40)
    print("⚠️ May have Windows compatibility issues")
    print("⚠️ Depends on numba/llvmlite which can be problematic")
    
    choice = input("\nDo you want to try original whisper anyway? (y/n): ").lower()
    if choice in ['y', 'yes']:
        if install_package("openai-whisper>=20231117", "OpenAI Whisper"):
            if test_import("whisper", "OpenAI Whisper"):
                print("\n✅ OpenAI Whisper installed successfully!")
                print("\n📝 Your voice agent will use openai-whisper for transcription.")
                return
    
    # No transcription available
    print("\n" + "="*60)
    print("⚠️ No transcription method was successfully installed")
    print("="*60)
    print("Don't worry! Your voice agent will still work, but without transcription features.")
    print("\nWhat this means:")
    print("✅ Voice calls will work normally")
    print("✅ Audio recording will work")
    print("❌ Call transcripts will not be generated")
    print("❌ Text summaries of calls will not be available")
    print("\nTo try again later, run: python setup_transcription_windows.py")

if __name__ == "__main__":
    main()
