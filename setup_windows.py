#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Setup Script for VoIP Voice Agent
Configures the system for Windows without requiring Linux/WSL
"""

import os
import sys
import json
import subprocess
import socket
from pathlib import Path
from typing import Optional

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for Windows console
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    except:
        # Fallback: replace Unicode characters with ASCII equivalents
        pass

def safe_print(text):
    """Print text with Windows-safe encoding"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace Unicode characters with ASCII equivalents
        safe_text = (text
                    .replace('✅', '[OK]')
                    .replace('❌', '[ERROR]')
                    .replace('⚠️', '[WARNING]')
                    .replace('🐍', '[Python]')
                    .replace('🔧', '[Config]')
                    .replace('📦', '[Install]')
                    .replace('🎤', '[Audio]')
                    .replace('🌐', '[Network]')
                    .replace('🎉', '[Success]'))
        print(safe_text)

def get_local_ip():
    """Get the local IP address of this Windows machine"""
    try:
        # Connect to a remote host to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except:
        return "192.168.1.100"  # Fallback

def create_env_file():
    """Create .env file with default values"""
    env_content = f"""# Voice Agent Environment Configuration

# Google AI API Key (Required)
# Get from: https://ai.google.dev/
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
GEMINI_API_KEY=your_google_ai_studio_api_key_here

# MongoDB Configuration (Optional)
MONGODB_HOST=localhost
MONGODB_PORT=27017

# Voice Agent Server Configuration
VOICE_AGENT_HOST=0.0.0.0
VOICE_AGENT_PORT=8000

# Windows Network Configuration
WINDOWS_LOCAL_IP={get_local_ip()}

# Gate VoIP Configuration
GATE_VOIP_IP=192.168.50.127
GATE_SIP_PORT=5060

# Phone Number (your SIM card number)
PHONE_NUMBER=+359898995151

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_DEBUG_LOGGING=false

# Audio Configuration
AUDIO_INPUT_DEVICE_INDEX=0
AUDIO_OUTPUT_DEVICE_INDEX=0
ENABLE_NOISE_REDUCTION=true
ENABLE_ECHO_CANCELLATION=true
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        safe_print(f"✅ Created .env file")
        safe_print("⚠️  Please edit .env and add your Google API key!")
    else:
        safe_print("⚠️  .env file already exists, skipping...")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        safe_print("❌ Python 3.8 or higher is required")
        return False
    safe_print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    try:
        safe_print("📦 Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_windows.txt"
        ])
        safe_print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        safe_print(f"❌ Failed to install dependencies: {e}")
        return False

def test_audio_devices():
    """Test available audio devices"""
    try:
        import pyaudio
        
        safe_print("\n🎤 Available Audio Devices:")
        pa = pyaudio.PyAudio()
        
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                safe_print(f"  Input Device {i}: {device_info['name']}")
        
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                safe_print(f"  Output Device {i}: {device_info['name']}")
        
        pa.terminate()
        
    except ImportError:
        safe_print("⚠️  PyAudio not installed, skipping audio device test")
    except Exception as e:
        safe_print(f"⚠️  Error testing audio devices: {e}")

def check_network_connectivity():
    """Test network connectivity to Gate VoIP system"""
    try:
        gate_ip = "192.168.50.127"  # From config
        sip_port = 5060
        
        safe_print(f"\n🌐 Testing network connectivity to Gate VoIP ({gate_ip})...")
        
        # Test ping
        result = subprocess.run(['ping', '-n', '1', gate_ip], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            safe_print(f"✅ Can reach Gate VoIP system at {gate_ip}")
            
            # Test SIP port
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            try:
                sock.connect((gate_ip, sip_port))
                safe_print(f"✅ SIP port {sip_port} is accessible")
            except:
                safe_print(f"⚠️  SIP port {sip_port} may not be accessible")
            finally:
                sock.close()
                
        else:
            safe_print(f"❌ Cannot reach Gate VoIP system at {gate_ip}")
            safe_print("   Please check your network connection and IP address")
        
    except Exception as e:
        safe_print(f"⚠️  Network test failed: {e}")

def update_config_with_detected_ip():
    """Update asterisk_config.json with detected local IP"""
    try:
        local_ip = get_local_ip()
        config_path = Path('asterisk_config.json')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['local_ip'] = local_ip
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            safe_print(f"✅ Updated config with local IP: {local_ip}")
        
    except Exception as e:
        safe_print(f"⚠️  Could not update config: {e}")

def create_startup_script():
    """Create a Windows startup script"""
    startup_script = """@echo off
echo Starting Windows VoIP Voice Agent...
echo.

REM Change to script directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found!
    echo Please run setup_windows.py first
    pause
    exit /b 1
)

REM Start the voice agent
echo Starting Voice Agent Server...
python windows_voice_agent.py --host 0.0.0.0 --port 8000

pause
"""
    
    with open('start_voice_agent.bat', 'w') as f:
        f.write(startup_script)
    
    safe_print("✅ Created start_voice_agent.bat")

def main():
    """Main setup function"""
    safe_print("🖥️  Windows VoIP Voice Agent Setup")
    safe_print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create environment file
    create_env_file()
    
    # Update config with detected IP
    update_config_with_detected_ip()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test audio devices
    test_audio_devices()
    
    # Test network connectivity
    check_network_connectivity()
    
    # Create startup script
    create_startup_script()
    
    safe_print("\n" + "=" * 50)
    safe_print("🎉 Setup Complete!")
    safe_print("=" * 50)
    safe_print("")
    safe_print("Next steps:")
    safe_print("1. Edit .env file and add your Google API key")
    safe_print("2. Verify Gate VoIP system configuration")
    safe_print("3. Run: python windows_voice_agent.py")
    safe_print("4. Or use: start_voice_agent.bat")
    safe_print("")
    safe_print("For testing:")
    safe_print("- Visit http://localhost:8000/health")
    safe_print("- Test call: curl -X POST http://localhost:8000/api/make_call -H \"Content-Type: application/json\" -d '{\"phone_number\": \"+1234567890\"}'")
    safe_print("")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        safe_print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
