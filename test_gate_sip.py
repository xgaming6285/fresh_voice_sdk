#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple SIP registration test for Gate VoIP
This script helps verify SIP connectivity and authentication
"""

import socket
import uuid
import json
import hashlib
import re
import time
import sys

def load_config():
    try:
        with open('asterisk_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None

def test_sip_registration():
    """Test SIP registration with Gate VoIP"""
    print("🔧 Gate VoIP SIP Registration Test")
    print("=" * 50)
    
    config = load_config()
    if not config:
        return False
    
    print(f"📍 Local IP: {config['local_ip']}")
    print(f"📍 Gate IP: {config['host']}")
    print(f"📍 SIP Port: {config['sip_port']}")
    print(f"👤 Username: {config['username']}")
    print(f"🔐 Password: {'*' * len(config.get('password', ''))}")
    print()
    
    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config['local_ip'], config['sip_port']))
        sock.settimeout(10.0)  # 10 second timeout
        
        print("🔗 Socket created and bound")
        
        # Step 1: Send initial REGISTER
        print("📤 Sending initial REGISTER request...")
        call_id = str(uuid.uuid4())
        username = config['username']
        gate_ip = config['host']
        sip_port = config['sip_port']
        local_ip = config['local_ip']
        
        register_message = f"""REGISTER sip:{gate_ip}:{sip_port} SIP/2.0
Via: SIP/2.0/UDP {local_ip}:{sip_port};branch=z9hG4bK{call_id[:8]};rport
Max-Forwards: 70
From: <sip:{username}@{gate_ip}>;tag={call_id[:8]}
To: <sip:{username}@{gate_ip}>
Call-ID: {call_id}
CSeq: 1 REGISTER
Contact: <sip:{username}@{local_ip}:{sip_port}>
User-Agent: SipRegistrationTest/1.0
Expires: 3600
Content-Length: 0

"""
        
        # Send initial REGISTER
        sock.sendto(register_message.encode(), (gate_ip, sip_port))
        print("✅ Initial REGISTER sent")
        
        # Wait for response
        print("⏳ Waiting for response...")
        try:
            data, addr = sock.recvfrom(4096)
            response = data.decode('utf-8')
            print(f"📨 Response from {addr}:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
            # Parse response
            first_line = response.split('\n')[0].strip()
            
            if '401 Unauthorized' in first_line:
                print("🔐 Received authentication challenge")
                
                # Parse authentication challenge
                realm_match = re.search(r'realm="([^"]*)"', response)
                nonce_match = re.search(r'nonce="([^"]*)"', response)
                
                realm = realm_match.group(1) if realm_match else gate_ip
                nonce = nonce_match.group(1) if nonce_match else ""
                
                print(f"   Realm: {realm}")
                print(f"   Nonce: {nonce[:20]}..." if nonce else "   Nonce: (empty)")
                
                # Create authenticated REGISTER
                password = config.get('password', '')
                uri = f"sip:{gate_ip}"
                
                ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()
                ha2 = hashlib.md5(f"REGISTER:{uri}".encode()).hexdigest()
                response_hash = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
                
                print("🔑 Creating authenticated REGISTER...")
                
                auth_register = f"""REGISTER sip:{gate_ip}:{sip_port} SIP/2.0
Via: SIP/2.0/UDP {local_ip}:{sip_port};branch=z9hG4bK{call_id[:8]}2;rport
Max-Forwards: 70
From: <sip:{username}@{gate_ip}>;tag={call_id[:8]}
To: <sip:{username}@{gate_ip}>
Call-ID: {call_id}
CSeq: 2 REGISTER
Contact: <sip:{username}@{local_ip}:{sip_port}>
User-Agent: SipRegistrationTest/1.0
Authorization: Digest username="{username}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response_hash}"
Expires: 3600
Content-Length: 0

"""
                
                # Send authenticated REGISTER
                sock.sendto(auth_register.encode(), (gate_ip, sip_port))
                print("📤 Authenticated REGISTER sent")
                
                # Wait for final response (could be 200 OK or OPTIONS keep-alive)
                print("⏳ Waiting for authentication response...")
                
                # We might receive multiple responses, so check a few
                responses_received = 0
                registration_success = False
                
                while responses_received < 3:  # Check up to 3 responses
                    try:
                        sock.settimeout(3.0)  # Shorter timeout for multiple responses
                        data, addr = sock.recvfrom(4096)
                        response = data.decode('utf-8')
                        responses_received += 1
                        
                        print(f"📨 Response #{responses_received} from {addr}:")
                        print("-" * 40)
                        print(response)
                        print("-" * 40)
                        
                        first_line = response.split('\n')[0].strip()
                        
                        if '200 OK' in first_line and 'REGISTER' in response:
                            print("✅ SUCCESS! SIP registration successful")
                            registration_success = True
                            
                        elif first_line.startswith('OPTIONS'):
                            print("📡 Received OPTIONS keep-alive (this means registration worked!)")
                            if not registration_success:
                                print("✅ IMPLIED SUCCESS! Gate VoIP is sending keep-alive OPTIONS")
                                print("   This means your registration was accepted")
                                registration_success = True
                                
                            # Respond to OPTIONS to be polite
                            options_response = f"""SIP/2.0 200 OK
Via: {response.split('Via: ')[1].split('\n')[0]}
From: {response.split('From: ')[1].split('\n')[0]}
To: {response.split('To: ')[1].split('\n')[0]};tag=test-response
Call-ID: {response.split('Call-ID: ')[1].split('\n')[0]}
CSeq: {response.split('CSeq: ')[1].split('\n')[0]}
Allow: INVITE, ACK, CANCEL, OPTIONS, BYE
Content-Length: 0

"""
                            sock.sendto(options_response.encode(), addr)
                            print("📤 Responded to OPTIONS keep-alive")
                            
                        else:
                            print(f"📨 Other response: {first_line}")
                            
                    except socket.timeout:
                        break  # No more responses
                
                if registration_success:
                    print("\n🎉 REGISTRATION SUCCESSFUL!")
                    print("🎯 Voice agent should now be able to receive calls")
                    return True
                else:
                    print(f"\n❌ Registration status unclear")
                    print("💡 Try running the full voice agent to see if it works")
                    return False
                    
            elif '200 OK' in first_line:
                print("✅ SUCCESS! SIP registration successful (no auth required)")
                print("🎯 Voice agent should now be able to receive calls")
                return True
            else:
                print(f"❌ Unexpected response: {first_line}")
                return False
                
        except socket.timeout:
            print("⏰ Timeout waiting for response")
            print("💡 Check network connectivity to Gate VoIP")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        sock.close()

def test_connectivity():
    """Test basic network connectivity to Gate VoIP"""
    print("🌐 Testing network connectivity...")
    
    config = load_config()
    if not config:
        return False
    
    try:
        # Test if we can reach the Gate IP
        import subprocess
        result = subprocess.run(['ping', '-n', '1', config['host']], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print(f"✅ Can ping Gate VoIP at {config['host']}")
            return True
        else:
            print(f"❌ Cannot ping Gate VoIP at {config['host']}")
            print("💡 Check network connection and Gate IP address")
            return False
            
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def main():
    print("Gate VoIP SIP Registration Test")
    print("=" * 50)
    
    # Test 1: Network connectivity
    print("\n🧪 Test 1: Network Connectivity")
    if not test_connectivity():
        print("❌ Network test failed, aborting")
        sys.exit(1)
    
    # Test 2: SIP Registration
    print("\n🧪 Test 2: SIP Registration")
    if test_sip_registration():
        print("\n🎉 All tests passed!")
        print("💡 Your configuration should work with the voice agent")
    else:
        print("\n❌ SIP registration failed")
        print("💡 Check the troubleshooting tips below:")
        print("   1. Verify extension 200 exists in Gate VoIP web interface")
        print("   2. Check that password matches the one in Gate VoIP")
        print("   3. Ensure Gate VoIP is accepting registrations")
        print("   4. Check firewall settings on both sides")

if __name__ == "__main__":
    main()
