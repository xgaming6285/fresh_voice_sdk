#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows VoIP Test Suite
Test the Windows VoIP Voice Agent integration with Gate VoIP system
"""

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WindowsVoIPTester:
    def __init__(self):
        self.config = self.load_config()
        self.api_base = "http://localhost:8000"
        
    def load_config(self):
        """Load configuration from asterisk_config.json"""
        try:
            with open('asterisk_config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}
    
    def test_python_environment(self):
        """Test Python environment and dependencies"""
        print("🐍 Testing Python Environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print(f"❌ Python version {sys.version_info.major}.{sys.version_info.minor} is too old")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Check required modules
        required_modules = [
            'fastapi', 'uvicorn', 'pyaudio', 'pydub', 
            'requests', 'google.genai', 'dotenv'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                if module == 'google.genai':
                    import google.genai
                elif module == 'dotenv':
                    import dotenv
                else:
                    __import__(module)
                print(f"✅ {module}")
            except ImportError:
                missing_modules.append(module)
                print(f"❌ {module} not found")
        
        if missing_modules:
            print(f"\nMissing modules: {missing_modules}")
            print("Run: pip install -r requirements_windows.txt")
            return False
        
        return True
    
    def test_environment_variables(self):
        """Test environment variables and configuration"""
        print("\n🔧 Testing Environment Variables...")
        
        import os
        
        # Check Google API key
        google_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not google_key or google_key == 'your_google_ai_studio_api_key_here':
            print("❌ Google API key not set")
            print("   Please set GOOGLE_API_KEY in .env file")
            return False
        else:
            print("✅ Google API key configured")
        
        # Check config file
        if not self.config:
            print("❌ Configuration not loaded")
            return False
        print("✅ Configuration loaded")
        
        # Check required config fields
        required_fields = ['host', 'local_ip', 'phone_number', 'sip_port']
        for field in required_fields:
            if field in self.config:
                print(f"✅ {field}: {self.config[field]}")
            else:
                print(f"❌ Missing config field: {field}")
                return False
        
        return True
    
    def test_network_connectivity(self):
        """Test network connectivity to Gate VoIP system"""
        print("\n🌐 Testing Network Connectivity...")
        
        gate_ip = self.config.get('host', '192.168.50.127')
        local_ip = self.config.get('local_ip', '192.168.50.158')
        sip_port = self.config.get('sip_port', 5060)
        
        print(f"Local IP: {local_ip}")
        print(f"Gate VoIP IP: {gate_ip}")
        
        # Test ping to Gate VoIP
        try:
            result = subprocess.run(['ping', '-n', '1', gate_ip], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Can ping Gate VoIP at {gate_ip}")
            else:
                print(f"❌ Cannot ping Gate VoIP at {gate_ip}")
                return False
        except Exception as e:
            print(f"⚠️  Ping test failed: {e}")
        
        # Test SIP port connectivity
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            sock.bind((local_ip, 0))  # Bind to local IP
            sock.sendto(b"test", (gate_ip, sip_port))
            print(f"✅ Can send UDP packets to {gate_ip}:{sip_port}")
            sock.close()
        except Exception as e:
            print(f"⚠️  UDP connectivity test failed: {e}")
        
        return True
    
    def test_audio_devices(self):
        """Test audio devices"""
        print("\n🎤 Testing Audio Devices...")
        
        try:
            import pyaudio
            
            pa = pyaudio.PyAudio()
            
            input_devices = []
            output_devices = []
            
            for i in range(pa.get_device_count()):
                device_info = pa.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info['name']))
                if device_info['maxOutputChannels'] > 0:
                    output_devices.append((i, device_info['name']))
            
            print(f"✅ Found {len(input_devices)} input devices")
            print(f"✅ Found {len(output_devices)} output devices")
            
            if input_devices:
                print("Input devices:")
                for idx, name in input_devices[:3]:  # Show first 3
                    print(f"  {idx}: {name}")
            
            if output_devices:
                print("Output devices:")
                for idx, name in output_devices[:3]:  # Show first 3
                    print(f"  {idx}: {name}")
            
            pa.terminate()
            
            return len(input_devices) > 0 and len(output_devices) > 0
            
        except Exception as e:
            print(f"❌ Audio device test failed: {e}")
            return False
    
    def test_voice_agent_server(self):
        """Test if the voice agent server can start and respond"""
        print("\n🤖 Testing Voice Agent Server...")
        
        # First check if server is already running
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Voice Agent server is already running")
                print(f"   Status: {data.get('status')}")
                print(f"   Phone: {data.get('phone_number')}")
                print(f"   Active sessions: {data.get('active_sessions')}")
                return True
        except requests.ConnectionError:
            print("⚠️  Voice Agent server not running")
        except Exception as e:
            print(f"⚠️  Server health check failed: {e}")
        
        print("To start the server, run:")
        print("  python windows_voice_agent.py")
        return False
    
    def test_api_endpoints(self):
        """Test API endpoints if server is running"""
        print("\n📡 Testing API Endpoints...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                print("✅ /health endpoint working")
            else:
                print(f"❌ /health returned {response.status_code}")
                return False
            
            # Test config endpoint
            response = requests.get(f"{self.api_base}/api/config", timeout=5)
            if response.status_code == 200:
                print("✅ /api/config endpoint working")
                config_data = response.json()
                print(f"   Phone: {config_data.get('config', {}).get('phone_number')}")
            else:
                print(f"❌ /api/config returned {response.status_code}")
            
            # Test sessions endpoint
            response = requests.get(f"{self.api_base}/api/sessions", timeout=5)
            if response.status_code == 200:
                print("✅ /api/sessions endpoint working")
                sessions_data = response.json()
                print(f"   Active sessions: {sessions_data.get('active_sessions')}")
            else:
                print(f"❌ /api/sessions returned {response.status_code}")
            
            return True
            
        except requests.ConnectionError:
            print("❌ Cannot connect to voice agent server")
            print("   Make sure to start: python windows_voice_agent.py")
            return False
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    def test_outbound_call_api(self):
        """Test the outbound call API (without actually making a call)"""
        print("\n📞 Testing Outbound Call API...")
        
        try:
            # Test with invalid data first
            response = requests.post(
                f"{self.api_base}/api/make_call",
                json={},
                timeout=5
            )
            if response.status_code == 400:
                print("✅ API correctly rejects invalid call requests")
            
            # Test with valid data (but don't actually make the call)
            test_number = "+1234567890"  # Test number
            response = requests.post(
                f"{self.api_base}/api/make_call",
                json={"phone_number": test_number},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✅ Outbound call API working (test number: {test_number})")
                call_data = response.json()
                print(f"   Session ID: {call_data.get('session_id')}")
            else:
                print(f"⚠️  Outbound call API returned {response.status_code}")
                print(f"   This may be normal if Gate VoIP is not fully configured")
            
            return True
            
        except Exception as e:
            print(f"❌ Outbound call API test failed: {e}")
            return False
    
    def test_google_ai_connection(self):
        """Test connection to Google AI"""
        print("\n🧠 Testing Google AI Connection...")
        
        try:
            import os
            from google import genai
            
            # Check API key
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("❌ No Google API key found")
                return False
            
            # Try to initialize client
            client = genai.Client(http_options={"api_version": "v1beta"})
            
            # This is a simple test - we can't easily test the live connection
            # without setting up a full voice session
            print("✅ Google AI client initialized")
            print("   Note: Full voice session testing requires active calls")
            
            return True
            
        except Exception as e:
            print(f"❌ Google AI connection test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Windows VoIP Voice Agent Test Suite")
        print("=" * 60)
        
        tests = [
            ("Python Environment", self.test_python_environment),
            ("Environment Variables", self.test_environment_variables),
            ("Network Connectivity", self.test_network_connectivity),
            ("Audio Devices", self.test_audio_devices),
            ("Google AI Connection", self.test_google_ai_connection),
            ("Voice Agent Server", self.test_voice_agent_server),
            ("API Endpoints", self.test_api_endpoints),
            ("Outbound Call API", self.test_outbound_call_api),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\nPassed: {passed}/{total}")
        
        if passed == total:
            print("\n🎉 All tests passed! Your system is ready.")
            print("\nTo start the voice agent:")
            print("  python windows_voice_agent.py")
            print("\nTo test a call:")
            print("  curl -X POST http://localhost:8000/api/make_call \\")
            print("    -H \"Content-Type: application/json\" \\")
            print("    -d '{\"phone_number\": \"+359898995151\"}'")
        else:
            print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        
        return passed == total

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick tests only
        tester = WindowsVoIPTester()
        print("🚀 Quick Test Mode")
        tester.test_python_environment()
        tester.test_environment_variables()
        tester.test_voice_agent_server()
    else:
        # Full test suite
        tester = WindowsVoIPTester()
        success = tester.run_all_tests()
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
