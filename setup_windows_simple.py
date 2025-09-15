#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Windows Setup Script for VoIP Voice Agent
Avoids Unicode characters for better Windows console compatibility
"""

import os
import sys
import json
import subprocess
import socket
from pathlib import Path

def get_local_ip():
    """Get the local IP address of this Windows machine"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except:
        return "192.168.50.158"  # Default fallback

def create_env_file():
    """Create .env file with default values"""
    local_ip = get_local_ip()
    env_content = f"""# Voice Agent Environment Configuration

# Google AI API Key (Required)
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
GEMINI_API_KEY=your_google_ai_studio_api_key_here

# MongoDB Configuration (Optional)
MONGODB_HOST=localhost
MONGODB_PORT=27017

# Voice Agent Server Configuration
VOICE_AGENT_HOST=0.0.0.0
VOICE_AGENT_PORT=8000

# Windows Network Configuration
WINDOWS_LOCAL_IP={local_ip}

# Gate VoIP Configuration
GATE_VOIP_IP=192.168.50.127
GATE_SIP_PORT=5060

# Phone Number
PHONE_NUMBER=+359898995151

# Logging Configuration
LOG_LEVEL=INFO
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("[OK] Created .env file")
        print("[WARNING] Please edit .env and add your Google API key!")
    else:
        print("[INFO] .env file already exists, skipping...")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        return False
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    try:
        print("[INFO] Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_windows.txt"
        ])
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False

def test_network():
    """Test network connectivity to Gate VoIP system"""
    try:
        gate_ip = "192.168.50.127"
        print(f"[INFO] Testing connectivity to Gate VoIP ({gate_ip})...")
        
        result = subprocess.run(['ping', '-n', '1', gate_ip], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"[OK] Can reach Gate VoIP system at {gate_ip}")
        else:
            print(f"[WARNING] Cannot reach Gate VoIP system at {gate_ip}")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Network test failed: {e}")
        return True  # Don't fail setup for network issues

def update_config():
    """Update config with detected local IP"""
    try:
        local_ip = get_local_ip()
        config_path = Path('asterisk_config.json')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['local_ip'] = local_ip
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"[OK] Updated config with local IP: {local_ip}")
        
    except Exception as e:
        print(f"[WARNING] Could not update config: {e}")

def main():
    """Main setup function"""
    print("Windows VoIP Voice Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create environment file
    create_env_file()
    
    # Update config
    update_config()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test network
    test_network()
    
    print("")
    print("=" * 50)
    print("[SUCCESS] Setup Complete!")
    print("=" * 50)
    print("")
    print("Next steps:")
    print("1. Edit .env file and add your Google API key")
    print("2. Run: python windows_voice_agent.py")
    print("")
    print("For testing:")
    print("- Visit http://localhost:8000/health")
    print("")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ERROR] Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)
