#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple SIP Test Script for Gate VoIP Integration
This script listens on port 5060 and logs all SIP messages to debug the integration
"""

import socket
import threading
import time
import json
from datetime import datetime

# Load configuration
with open('asterisk_config.json', 'r') as f:
    config = json.load(f)

LOCAL_IP = config['local_ip']  # Your Windows IP
GATE_IP = config['host']       # Gate VoIP IP
SIP_PORT = 5060

class SIPTestListener:
    def __init__(self):
        self.running = False
        self.socket = None
        
    def start(self):
        """Start SIP listener for testing"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((LOCAL_IP, SIP_PORT))
            self.running = True
            
            print(f"[INFO] SIP Test Listener started with FULL LOGGING")
            print(f"[INFO] Listening on {LOCAL_IP}:{SIP_PORT}")
            print(f"[INFO] Expecting messages from Gate VoIP at {GATE_IP}")
            print(f"[INFO] Phone number: {config['phone_number']}")
            print("=" * 80)
            print("LOGGING FORMAT:")
            print("  <<<< RECEIVED = Incoming SIP messages")
            print("  >>>> SENDING  = Outgoing SIP responses")
            print("=" * 80)
            print("Now try calling +359898995151 to test the connection...")
            print("=" * 80)
            print("Waiting for SIP messages...\n")
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(4096)
                    message = data.decode('utf-8', errors='ignore')
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] <<<< RECEIVED SIP MESSAGE <<<<")
                    print(f"From: {addr[0]}:{addr[1]}")
                    print(f"Size: {len(data)} bytes")
                    print("=" * 60)
                    print(message)
                    print("=" * 60)
                    
                    # Basic SIP response for testing
                    if "INVITE" in message:
                        print("\n[RESPONSE] Processing INVITE request...")
                        response = self.create_basic_ok_response(message, addr)
                        response_bytes = response.encode()
                        
                        print(f"[{timestamp}] >>>> SENDING SIP RESPONSE >>>>")
                        print(f"To: {addr[0]}:{addr[1]}")
                        print(f"Size: {len(response_bytes)} bytes")
                        print("=" * 60)
                        print(response)
                        print("=" * 60)
                        
                        self.socket.sendto(response_bytes, addr)
                        print(f"[{timestamp}] Response sent successfully!")
                        
                    elif "BYE" in message:
                        print("\n[RESPONSE] Processing BYE request...")
                        response = "SIP/2.0 200 OK\r\n\r\n"
                        response_bytes = response.encode()
                        
                        print(f"[{timestamp}] >>>> SENDING BYE RESPONSE >>>>")
                        print(f"To: {addr[0]}:{addr[1]}")
                        print(f"Size: {len(response_bytes)} bytes")
                        print("=" * 60)
                        print(response)
                        print("=" * 60)
                        
                        self.socket.sendto(response_bytes, addr)
                        print(f"[{timestamp}] BYE response sent successfully!")
                        
                    elif "ACK" in message:
                        print("\n[INFO] *** ACK RECEIVED - CALL ESTABLISHED! ***")
                        print(f"[{timestamp}] Call session is now active")
                        
                    elif "CANCEL" in message:
                        print("\n[RESPONSE] Processing CANCEL request...")
                        response = "SIP/2.0 200 OK\r\n\r\n"
                        response_bytes = response.encode()
                        
                        print(f"[{timestamp}] >>>> SENDING CANCEL RESPONSE >>>>")
                        print(f"To: {addr[0]}:{addr[1]}")
                        print(f"Size: {len(response_bytes)} bytes")
                        print("=" * 60)
                        print(response)
                        print("=" * 60)
                        
                        self.socket.sendto(response_bytes, addr)
                        print(f"[{timestamp}] CANCEL response sent successfully!")
                        print(f"[{timestamp}] *** CALL CANCELLED ***")
                        
                    elif "OPTIONS" in message:
                        print("\n[RESPONSE] Processing OPTIONS request...")
                        response = self.create_basic_ok_response(message, addr)
                        response_bytes = response.encode()
                        
                        print(f"[{timestamp}] >>>> SENDING OPTIONS RESPONSE >>>>")
                        print(f"To: {addr[0]}:{addr[1]}")
                        print(f"Size: {len(response_bytes)} bytes")
                        print("=" * 60)
                        print(response)
                        print("=" * 60)
                        
                        self.socket.sendto(response_bytes, addr)
                        print(f"[{timestamp}] OPTIONS response sent successfully!")
                        
                    else:
                        print(f"\n[INFO] Unknown SIP message type received")
                        print(f"[{timestamp}] Message not handled - no response sent")
                    
                except Exception as e:
                    if self.running:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n[{timestamp}] [ERROR] Error processing SIP message: {e}")
                        print(f"[ERROR] From: {addr if 'addr' in locals() else 'Unknown'}")
        except Exception as e:
            print(f"[ERROR] Failed to start SIP listener: {e}")
            return False
            
    def create_basic_ok_response(self, invite_message, addr):
        """Create a basic SIP 200 OK response"""
        # Extract some basic info from INVITE
        call_id = "test-call-123"
        cseq = "1 INVITE"  # default fallback
        
        for line in invite_message.split('\n'):
            if line.startswith('Call-ID:'):
                call_id = line.split(':', 1)[1].strip()
            elif line.startswith('CSeq:'):
                cseq = line.split(':', 1)[1].strip()
                
        response = f"""SIP/2.0 200 OK
Via: SIP/2.0/UDP {addr[0]}:{addr[1]}
From: <sip:caller@{addr[0]}>
To: <sip:voice-agent@{LOCAL_IP}>
Call-ID: {call_id}
CSeq: {cseq}
Contact: <sip:voice-agent@{LOCAL_IP}:{SIP_PORT}>
Content-Length: 0

"""
        return response
        
    def stop(self):
        """Stop the SIP listener"""
        self.running = False
        if self.socket:
            self.socket.close()
            
def test_network_connectivity():
    """Test basic network connectivity to Gate VoIP"""
    print(f"[TEST] Testing network connectivity to Gate VoIP ({GATE_IP})...")
    
    try:
        # Test ping
        import subprocess
        result = subprocess.run(['ping', '-n', '1', GATE_IP], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print(f"[OK] Can ping Gate VoIP at {GATE_IP}")
        else:
            print(f"[WARNING] Cannot ping Gate VoIP at {GATE_IP}")
            
    except Exception as e:
        print(f"[WARNING] Ping test failed: {e}")
    
    # Test UDP socket binding
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_socket.bind((LOCAL_IP, SIP_PORT))
        test_socket.close()
        print(f"[OK] Can bind to {LOCAL_IP}:{SIP_PORT}")
    except Exception as e:
        print(f"[ERROR] Cannot bind to {LOCAL_IP}:{SIP_PORT} - {e}")
        return False
        
    return True

def main():
    """Main test function"""
    print("SIP Connectivity Test for Gate VoIP Integration")
    print("=" * 60)
    print(f"Local IP (Windows): {LOCAL_IP}")
    print(f"Gate VoIP IP: {GATE_IP}")
    print(f"SIP Port: {SIP_PORT}")
    print(f"Phone Number: {config['phone_number']}")
    print("=" * 60)
    
    # Test network connectivity first
    if not test_network_connectivity():
        print("\n[ERROR] Network connectivity test failed!")
        print("Please check:")
        print("1. Your Windows IP address is correct")
        print("2. Gate VoIP system is accessible")
        print("3. Windows Firewall allows UDP port 5060")
        return
    
    # Start SIP listener
    listener = SIPTestListener()
    
    try:
        print("\n[INFO] Starting SIP test listener...")
        print("Press Ctrl+C to stop")
        listener.start()
        
    except KeyboardInterrupt:
        print("\n[INFO] Stopping SIP test listener...")
        listener.stop()
        
    except Exception as e:
        print(f"\n[ERROR] SIP test failed: {e}")
        
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
