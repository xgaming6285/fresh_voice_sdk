#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Agent AGI Script for Asterisk
This script is called by Asterisk's AGI interface to handle voice agent calls.
It communicates with the AGI voice server to process audio and manage the conversation.
"""

import sys
import requests
import json
import base64
import logging
import time
import wave
import io
from typing import Optional

# Configure logging
logging.basicConfig(
    filename='/var/log/asterisk/voice_agent_agi.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AsteriskAGI:
    """Simple AGI interface for Asterisk communication"""
    
    def __init__(self):
        self.env = {}
        self.session_id = None
        self.voice_server_url = "http://localhost:8000"
        
        # Read AGI environment
        self._read_agi_env()
    
    def _read_agi_env(self):
        """Read AGI environment variables from stdin"""
        while True:
            line = sys.stdin.readline().strip()
            if line == '':
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
        
        logger.info(f"AGI Environment: {self.env}")
    
    def _send_command(self, command: str) -> str:
        """Send command to Asterisk and get response"""
        sys.stdout.write(command + '\n')
        sys.stdout.flush()
        response = sys.stdin.readline().strip()
        logger.debug(f"Command: {command} | Response: {response}")
        return response
    
    def answer(self):
        """Answer the call"""
        return self._send_command("ANSWER")
    
    def hangup(self):
        """Hang up the call"""
        return self._send_command("HANGUP")
    
    def say_text(self, text: str, escape_digits: str = ""):
        """Say text using text-to-speech"""
        return self._send_command(f'SAY TEXT "{text}" "{escape_digits}"')
    
    def stream_file(self, filename: str, escape_digits: str = ""):
        """Stream an audio file"""
        return self._send_command(f'STREAM FILE "{filename}" "{escape_digits}"')
    
    def wait_for_digit(self, timeout: int = 5000):
        """Wait for DTMF digit"""
        return self._send_command(f"WAIT FOR DIGIT {timeout}")
    
    def record_file(self, filename: str, format: str = "wav", escape_digits: str = "#", timeout: int = 10000):
        """Record audio from the caller"""
        return self._send_command(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout}')
    
    def get_variable(self, variable: str):
        """Get channel variable value"""
        response = self._send_command(f"GET VARIABLE {variable}")
        # Parse response: 200 result=1 (value)
        if "result=1" in response:
            # Extract value from parentheses
            start = response.find('(') + 1
            end = response.find(')')
            if start > 0 and end > start:
                return response[start:end]
        return None
    
    def set_variable(self, variable: str, value: str):
        """Set channel variable"""
        return self._send_command(f'SET VARIABLE "{variable}" "{value}"')

class VoiceAgentAGIHandler:
    """Handles the voice agent conversation through AGI"""
    
    def __init__(self, agi: AsteriskAGI, session_id: str):
        self.agi = agi
        self.session_id = session_id
        self.voice_server_url = "http://localhost:8000"
        self.caller_id = agi.env.get('agi_callerid', 'Unknown')
        self.called_number = agi.env.get('agi_extension', '')
        self.channel = agi.env.get('agi_channel', '')
        
        logger.info(f"Starting voice agent session {session_id} for {self.caller_id} -> {self.called_number}")
    
    def initialize_session(self) -> bool:
        """Initialize the voice agent session on the server"""
        try:
            response = requests.post(
                f"{self.voice_server_url}/agi/new_call",
                json={
                    "caller_id": self.caller_id,
                    "called_number": self.called_number,
                    "channel": self.channel,
                    "session_id": self.session_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Voice agent session {self.session_id} initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize session: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing voice agent session: {e}")
            return False
    
    def handle_conversation(self):
        """Main conversation loop"""
        try:
            # Initialize voice session
            if not self.initialize_session():
                self.agi.say_text("Sorry, there was a technical problem. Please try again later.")
                return
            
            # Play initial greeting
            self.agi.say_text("Hello! Please speak after the tone, and I will respond to you.")
            
            # Conversation loop
            conversation_timeout = 0
            max_idle_time = 300  # 5 minutes max idle
            
            while conversation_timeout < max_idle_time:
                try:
                    # Record user input
                    timestamp = str(int(time.time()))
                    record_file = f"/tmp/voice_input_{self.session_id}_{timestamp}"
                    
                    # Record with timeout and # to stop
                    result = self.agi.record_file(record_file, "wav", "#*0", 10000)  # 10 second timeout
                    
                    if "timeout" in result.lower():
                        conversation_timeout += 10
                        continue
                    
                    # Check for special DTMF commands
                    if "dtmf" in result.lower():
                        digit = self.extract_dtmf(result)
                        if digit:
                            dtmf_response = self.handle_dtmf(digit)
                            if dtmf_response.get("action") == "hangup":
                                break
                            elif dtmf_response.get("action") == "transfer":
                                self.agi.say_text("Transferring you to an operator. Please hold.")
                                # Add transfer logic here
                                break
                            continue
                    
                    # Process recorded audio through voice agent
                    response_text, response_audio = self.process_recorded_audio(record_file + ".wav")
                    
                    if response_text:
                        # Play the AI response
                        self.agi.say_text(response_text)
                    
                    # Reset timeout on successful interaction
                    conversation_timeout = 0
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    self.agi.say_text("I'm having trouble understanding. Could you please repeat that?")
                    conversation_timeout += 5
            
            # End conversation
            self.agi.say_text("Thank you for calling. Have a great day!")
            
        except Exception as e:
            logger.error(f"Error in conversation handling: {e}")
            self.agi.say_text("I apologize, but I'm having technical difficulties. Please call back later.")
        
        finally:
            # Clean up session
            self.cleanup_session()
    
    def extract_dtmf(self, result: str) -> Optional[str]:
        """Extract DTMF digit from AGI result"""
        try:
            # Look for dtmf= in the result
            if "dtmf=" in result:
                start = result.find("dtmf=") + 5
                return result[start:start+1]
        except:
            pass
        return None
    
    def handle_dtmf(self, digit: str) -> dict:
        """Handle DTMF input via voice server"""
        try:
            response = requests.post(
                f"{self.voice_server_url}/agi/dtmf/{self.session_id}",
                json={"digit": digit},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            
        except Exception as e:
            logger.error(f"Error handling DTMF {digit}: {e}")
        
        return {"action": "continue"}
    
    def process_recorded_audio(self, audio_file: str) -> tuple:
        """Process recorded audio through the voice agent"""
        try:
            # Read the recorded audio file
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Encode as base64 for transmission
            audio_b64 = base64.b64encode(audio_data).decode()
            
            # Send to voice server for processing
            response = requests.post(
                f"{self.voice_server_url}/agi/audio/{self.session_id}",
                json={"audio": audio_b64},
                timeout=30  # Allow time for AI processing
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Get response audio if available
                response_audio = None
                if result.get("audio"):
                    response_audio = base64.b64decode(result["audio"])
                
                # For now, we'll use text-to-speech instead of audio response
                # In a full implementation, you'd play the response audio directly
                
                return "I understand. Let me help you with that.", response_audio
            
            else:
                logger.error(f"Voice server error: {response.status_code} - {response.text}")
                return "I'm sorry, I didn't catch that. Could you please repeat?", None
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "I'm having trouble processing your request. Please try again.", None
        
        finally:
            # Clean up temporary file
            try:
                import os
                os.remove(audio_file)
            except:
                pass
    
    def cleanup_session(self):
        """Clean up the voice agent session"""
        try:
            requests.post(
                f"{self.voice_server_url}/agi/end_call/{self.session_id}",
                timeout=5
            )
            logger.info(f"Cleaned up session {self.session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

def main():
    """Main function called by Asterisk AGI"""
    try:
        # Initialize AGI
        agi = AsteriskAGI()
        
        # Get session ID from command line arguments
        session_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        
        # Answer the call
        agi.answer()
        
        # Create and start voice agent handler
        handler = VoiceAgentAGIHandler(agi, session_id)
        handler.handle_conversation()
        
    except Exception as e:
        logger.error(f"Fatal error in AGI script: {e}")
        try:
            agi = AsteriskAGI()
            agi.say_text("Sorry, there was an error. Please try again later.")
        except:
            pass
    
    finally:
        logger.info("AGI script completed")

if __name__ == "__main__":
    main()