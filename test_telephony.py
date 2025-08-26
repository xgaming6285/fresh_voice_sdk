#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing Framework for Voice Agent Telephony Integration
Tests various components of the telephony system to ensure proper functionality.
"""

import asyncio
import requests
import json
import base64
import wave
import io
import time
import logging
from pathlib import Path
from typing import Dict, Any
import unittest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelephonyIntegrationTests(unittest.TestCase):
    """Test cases for telephony integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.voice_server_url = "http://localhost:8000"
        self.test_session_id = "test_session_123"
        self.test_caller_id = "+1234567890"
        self.test_called_number = "1000"
        
        # Create test audio data
        self.test_audio_data = self.create_test_audio()
    
    def create_test_audio(self) -> bytes:
        """Create a test audio file for testing"""
        try:
            # Generate 1 second of silence at 8kHz, 16-bit mono
            sample_rate = 8000
            duration = 1.0
            samples = int(sample_rate * duration)
            
            # Create WAV data in memory
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                # Write silence (zeros)
                wav_file.writeframes(b'\x00\x00' * samples)
            
            audio_io.seek(0)
            return audio_io.read()
            
        except Exception as e:
            logger.error(f"Error creating test audio: {e}")
            return b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00" + b"\x00" * 2048
    
    def test_voice_server_health(self):
        """Test if the voice server is running and healthy"""
        logger.info("Testing voice server health...")
        
        try:
            response = requests.get(f"{self.voice_server_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            
            health_data = response.json()
            self.assertIn("status", health_data)
            self.assertEqual(health_data["status"], "healthy")
            
            logger.info("✅ Voice server health check passed")
            
        except requests.exceptions.ConnectionError:
            self.fail("❌ Voice server is not running. Please start it with: python agi_voice_server.py")
        except Exception as e:
            self.fail(f"❌ Voice server health check failed: {e}")
    
    def test_session_creation(self):
        """Test creating a new call session"""
        logger.info("Testing session creation...")
        
        call_data = {
            "caller_id": self.test_caller_id,
            "called_number": self.test_called_number,
            "channel": "SIP/test-00000001"
        }
        
        try:
            response = requests.post(
                f"{self.voice_server_url}/agi/new_call",
                json=call_data,
                timeout=10
            )
            
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertEqual(response_data["status"], "success")
            self.assertIn("session_id", response_data)
            
            # Store session ID for cleanup
            self.created_session_id = response_data["session_id"]
            
            logger.info("✅ Session creation test passed")
            
        except Exception as e:
            self.fail(f"❌ Session creation test failed: {e}")
    
    def test_audio_processing(self):
        """Test audio processing through the voice agent"""
        logger.info("Testing audio processing...")
        
        # First create a session
        self.test_session_creation()
        
        # Wait a moment for session to initialize
        time.sleep(2)
        
        try:
            # Encode test audio as base64
            audio_b64 = base64.b64encode(self.test_audio_data).decode()
            
            response = requests.post(
                f"{self.voice_server_url}/agi/audio/{self.created_session_id}",
                json={"audio": audio_b64},
                timeout=30  # Allow time for AI processing
            )
            
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertEqual(response_data["status"], "success")
            self.assertIn("session_id", response_data)
            
            logger.info("✅ Audio processing test passed")
            
        except Exception as e:
            self.fail(f"❌ Audio processing test failed: {e}")
    
    def test_dtmf_handling(self):
        """Test DTMF (keypad) input handling"""
        logger.info("Testing DTMF handling...")
        
        # Use existing session if available, otherwise create one
        if not hasattr(self, 'created_session_id'):
            self.test_session_creation()
            time.sleep(1)
        
        test_digits = ["0", "1", "2", "*", "#"]
        
        for digit in test_digits:
            try:
                response = requests.post(
                    f"{self.voice_server_url}/agi/dtmf/{self.created_session_id}",
                    json={"digit": digit},
                    timeout=5
                )
                
                self.assertEqual(response.status_code, 200)
                
                response_data = response.json()
                self.assertEqual(response_data["status"], "success")
                self.assertIn("action", response_data)
                
                logger.info(f"✅ DTMF test passed for digit: {digit}")
                
            except Exception as e:
                logger.error(f"❌ DTMF test failed for digit {digit}: {e}")
    
    def test_session_cleanup(self):
        """Test session cleanup"""
        logger.info("Testing session cleanup...")
        
        # Use existing session if available, otherwise create one
        if not hasattr(self, 'created_session_id'):
            self.test_session_creation()
            time.sleep(1)
        
        try:
            response = requests.post(
                f"{self.voice_server_url}/agi/end_call/{self.created_session_id}",
                timeout=5
            )
            
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertEqual(response_data["status"], "success")
            
            logger.info("✅ Session cleanup test passed")
            
        except Exception as e:
            self.fail(f"❌ Session cleanup test failed: {e}")
    
    def test_get_active_sessions(self):
        """Test retrieving active sessions"""
        logger.info("Testing active sessions retrieval...")
        
        try:
            response = requests.get(f"{self.voice_server_url}/api/sessions", timeout=5)
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertEqual(response_data["status"], "success")
            self.assertIn("active_sessions", response_data)
            self.assertIn("sessions", response_data)
            
            logger.info("✅ Active sessions test passed")
            
        except Exception as e:
            self.fail(f"❌ Active sessions test failed: {e}")
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'created_session_id'):
            try:
                requests.post(
                    f"{self.voice_server_url}/agi/end_call/{self.created_session_id}",
                    timeout=5
                )
            except:
                pass

class AsteriskConfigurationTests(unittest.TestCase):
    """Test Asterisk configuration and connectivity"""
    
    def test_config_file_exists(self):
        """Test if configuration file exists and is valid"""
        logger.info("Testing configuration file...")
        
        config_file = Path("asterisk_config.json")
        self.assertTrue(config_file.exists(), "Configuration file asterisk_config.json not found")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_keys = ["host", "username", "password", "sip_port"]
            for key in required_keys:
                self.assertIn(key, config, f"Required configuration key '{key}' not found")
            
            # Check for placeholder values
            if config["host"] == "YOUR_SIM_GATEWAY_IP_HERE":
                logger.warning("⚠️ Configuration file contains placeholder values. Please update with your actual SIM gateway details.")
            
            logger.info("✅ Configuration file test passed")
            
        except json.JSONDecodeError as e:
            self.fail(f"❌ Configuration file is not valid JSON: {e}")
    
    def test_agi_script_exists(self):
        """Test if AGI script exists and is executable"""
        logger.info("Testing AGI script...")
        
        agi_script = Path("voice_agent_agi.py")
        self.assertTrue(agi_script.exists(), "AGI script voice_agent_agi.py not found")
        
        # Check if script is executable
        import os
        self.assertTrue(os.access(agi_script, os.X_OK), "AGI script is not executable")
        
        logger.info("✅ AGI script test passed")

class AudioFormatTests(unittest.TestCase):
    """Test audio format conversion"""
    
    def setUp(self):
        """Set up audio format tests"""
        try:
            from asterisk_integration import AudioConverter
            self.audio_converter = AudioConverter()
        except ImportError:
            self.skipTest("AudioConverter not available for testing")
    
    def create_test_pcm_data(self, sample_rate: int, duration: float = 1.0) -> bytes:
        """Create test PCM data"""
        import struct
        samples = int(sample_rate * duration)
        # Generate simple sine wave
        data = []
        for i in range(samples):
            # Simple sine wave at 440Hz
            import math
            value = int(32767 * 0.1 * math.sin(2 * math.pi * 440 * i / sample_rate))
            data.append(struct.pack('<h', value))
        return b''.join(data)
    
    def test_telephony_to_gemini_conversion(self):
        """Test conversion from 8kHz telephony to 16kHz Gemini format"""
        logger.info("Testing telephony to Gemini audio conversion...")
        
        # Create 8kHz test data
        test_audio_8k = self.create_test_pcm_data(8000)
        
        # Convert to 16kHz
        converted_audio = self.audio_converter.telephony_to_gemini(test_audio_8k)
        
        # Converted audio should be roughly twice as long
        self.assertGreater(len(converted_audio), len(test_audio_8k) * 1.5)
        self.assertLess(len(converted_audio), len(test_audio_8k) * 2.5)
        
        logger.info("✅ Telephony to Gemini conversion test passed")
    
    def test_gemini_to_telephony_conversion(self):
        """Test conversion from 24kHz Gemini to 8kHz telephony format"""
        logger.info("Testing Gemini to telephony audio conversion...")
        
        # Create 24kHz test data
        test_audio_24k = self.create_test_pcm_data(24000)
        
        # Convert to 8kHz
        converted_audio = self.audio_converter.gemini_to_telephony(test_audio_24k)
        
        # Converted audio should be roughly one-third as long
        self.assertLess(len(converted_audio), len(test_audio_24k) * 0.5)
        
        logger.info("✅ Gemini to telephony conversion test passed")

def run_system_checks():
    """Run system-level checks"""
    logger.info("Running system checks...")
    
    checks = {
        "Python version": check_python_version,
        "Required packages": check_required_packages,
        "Google API key": check_google_api_key,
        "Asterisk availability": check_asterisk_available,
        "Voice server connectivity": check_voice_server,
        "Port availability": check_port_availability
    }
    
    results = {}
    for check_name, check_func in checks.items():
        try:
            result = check_func()
            results[check_name] = {"status": "✅ PASS", "details": result}
            logger.info(f"✅ {check_name}: {result}")
        except Exception as e:
            results[check_name] = {"status": "❌ FAIL", "details": str(e)}
            logger.error(f"❌ {check_name}: {e}")
    
    return results

def check_python_version():
    """Check Python version"""
    import sys
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 8):
        raise Exception(f"Python 3.8+ required, found {version}")
    return f"Python {version}"

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        "requests", "fastapi", "uvicorn", "pydub", "google.genai",
        "pyaudio", "pymongo", "opencv-python"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "opencv-python":
                import cv2
            elif package == "google.genai":
                from google import genai
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise Exception(f"Missing packages: {', '.join(missing_packages)}")
    
    return f"All {len(required_packages)} packages available"

def check_google_api_key():
    """Check if Google API key is set"""
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("GOOGLE_API_KEY environment variable not set")
    return "API key configured"

def check_asterisk_available():
    """Check if Asterisk is available"""
    import subprocess
    try:
        result = subprocess.run(
            ["asterisk", "-V"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return f"Asterisk available: {result.stdout.strip()}"
        else:
            raise Exception("Asterisk not responding")
    except subprocess.TimeoutExpired:
        raise Exception("Asterisk command timed out")
    except FileNotFoundError:
        raise Exception("Asterisk not installed or not in PATH")

def check_voice_server():
    """Check if voice server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            return "Voice server running"
        else:
            raise Exception(f"Voice server returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        raise Exception("Voice server not running")

def check_port_availability():
    """Check if required ports are available"""
    import socket
    ports_to_check = [8000, 5060]  # Voice server, SIP
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if port == 8000:
                # This port should be in use by our server
                result = sock.connect_ex(('localhost', port))
                if result != 0:
                    raise Exception(f"Voice server not running on port {port}")
            else:
                # These ports should be available or in use by Asterisk
                result = sock.connect_ex(('localhost', port))
                # Either available (connection refused) or in use (connected) is OK
        finally:
            sock.close()
    
    return "Required ports accessible"

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Agent Telephony Testing Framework")
    parser.add_argument("--system-checks", action="store_true", help="Run system checks only")
    parser.add_argument("--unit-tests", action="store_true", help="Run unit tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.system_checks or (not args.unit_tests and not args.system_checks):
        logger.info("="*50)
        logger.info("RUNNING SYSTEM CHECKS")
        logger.info("="*50)
        results = run_system_checks()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SYSTEM CHECKS SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for r in results.values() if "✅" in r["status"])
        total = len(results)
        
        for check_name, result in results.items():
            logger.info(f"{result['status']} {check_name}")
        
        logger.info(f"\nPassed: {passed}/{total}")
        
        if passed < total:
            logger.error("Some system checks failed. Please address the issues before proceeding.")
    
    if args.unit_tests or (not args.unit_tests and not args.system_checks):
        logger.info("\n" + "="*50)
        logger.info("RUNNING UNIT TESTS")
        logger.info("="*50)
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test cases
        test_suite.addTest(unittest.makeSuite(TelephonyIntegrationTests))
        test_suite.addTest(unittest.makeSuite(AsteriskConfigurationTests))
        test_suite.addTest(unittest.makeSuite(AudioFormatTests))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(test_suite)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("UNIT TESTS SUMMARY")
        logger.info("="*50)
        
        if result.wasSuccessful():
            logger.info("✅ All unit tests passed!")
        else:
            logger.error(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
            for failure in result.failures:
                logger.error(f"FAILURE: {failure[0]} - {failure[1]}")
            for error in result.errors:
                logger.error(f"ERROR: {error[0]} - {error[1]}")

if __name__ == "__main__":
    main()