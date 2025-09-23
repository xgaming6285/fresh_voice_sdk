#!/usr/bin/env python3
"""
Test script for bidirectional voice agent calls
Tests both incoming call reception and outbound call initiation
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
VOICE_AGENT_API = "http://192.168.50.128:8000"  # Update with your voice agent IP
TEST_PHONE_NUMBER = "+359XXXXXXXXX"  # Replace with your test number

def test_health():
    """Test if the voice agent API is running"""
    try:
        response = requests.get(f"{VOICE_AGENT_API}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Voice Agent API is healthy")
            print(f"   Active Sessions: {data.get('active_sessions', 0)}")
            print(f"   Phone Number: {data.get('phone_number')}")
            print(f"   Local IP: {data.get('local_ip')}")
            print(f"   Gate VoIP: {data.get('gate_voip')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Cannot connect to voice agent API: {e}")
        return False

def test_configuration():
    """Test voice agent configuration"""
    try:
        response = requests.get(f"{VOICE_AGENT_API}/api/config", timeout=5)
        if response.status_code == 200:
            data = response.json()
            config = data.get('config', {})
            print("📋 Voice Agent Configuration:")
            print(f"   Phone Number: {config.get('phone_number')}")
            print(f"   Local IP: {config.get('local_ip')}")
            print(f"   Gate VoIP IP: {config.get('gate_voip_ip')}")
            print(f"   SIP Port: {config.get('sip_port')}")
            return True
        else:
            print(f"❌ Configuration check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Cannot get configuration: {e}")
        return False

def test_outbound_call(phone_number):
    """Test making an outbound call"""
    try:
        print(f"📞 Testing outbound call to {phone_number}...")
        
        payload = {"phone_number": phone_number}
        response = requests.post(
            f"{VOICE_AGENT_API}/api/make_call", 
            json=payload, 
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get('session_id')
            print(f"✅ Outbound call initiated successfully!")
            print(f"   Session ID: {session_id}")
            print(f"   Target Number: {data.get('phone_number')}")
            print(f"   Message: {data.get('message')}")
            return session_id
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            print(f"❌ Outbound call failed: {response.status_code}")
            print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Error making outbound call: {e}")
        return None

def get_active_sessions():
    """Get list of active call sessions"""
    try:
        response = requests.get(f"{VOICE_AGENT_API}/api/sessions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            sessions = data.get('sessions', [])
            active_count = data.get('active_sessions', 0)
            
            print(f"📊 Active Sessions: {active_count}")
            for session in sessions:
                print(f"   Session ID: {session.get('session_id')}")
                print(f"   Caller ID: {session.get('caller_id')}")
                print(f"   Called Number: {session.get('called_number')}")
                print(f"   Status: {session.get('status')}")
                print(f"   Duration: {session.get('duration_seconds'):.1f}s")
                print()
            
            return sessions
        else:
            print(f"❌ Cannot get sessions: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"❌ Error getting sessions: {e}")
        return []

def main():
    """Main test function"""
    print("=" * 60)
    print("🤖 Voice Agent Bidirectional Call Test")
    print("=" * 60)
    print()
    
    # Test 1: Health Check
    print("1️⃣ Testing API Health...")
    if not test_health():
        print("❌ Health check failed. Make sure the voice agent is running.")
        sys.exit(1)
    print()
    
    # Test 2: Configuration Check
    print("2️⃣ Testing Configuration...")
    if not test_configuration():
        print("❌ Configuration check failed.")
        sys.exit(1)
    print()
    
    # Test 3: Active Sessions (before test)
    print("3️⃣ Checking Active Sessions (before test)...")
    initial_sessions = get_active_sessions()
    print()
    
    # Test 4: Outbound Call Test
    print("4️⃣ Testing Outbound Call Capability...")
    
    if TEST_PHONE_NUMBER == "+359XXXXXXXXX":
        print("⚠️  Please update TEST_PHONE_NUMBER in the script with your actual test number")
        test_number = input("Enter test phone number (with country code, e.g., +359898123456): ").strip()
        if not test_number:
            print("❌ No test number provided, skipping outbound test")
            test_number = None
    else:
        test_number = TEST_PHONE_NUMBER
    
    if test_number:
        session_id = test_outbound_call(test_number)
        if session_id:
            print("⏳ Waiting 5 seconds for call to establish...")
            time.sleep(5)
            
            # Check sessions after call
            print("📊 Checking sessions after outbound call...")
            get_active_sessions()
        else:
            print("❌ Outbound call test failed")
    print()
    
    # Test 5: Incoming Call Instructions
    print("5️⃣ Testing Incoming Call Reception...")
    phone_number = None
    try:
        response = requests.get(f"{VOICE_AGENT_API}/api/config", timeout=5)
        if response.status_code == 200:
            config = response.json().get('config', {})
            phone_number = config.get('phone_number')
    except:
        pass
    
    if phone_number:
        print(f"📞 To test incoming calls, dial: {phone_number}")
        print("   You should:")
        print("   1. Hear a greeting message")
        print("   2. Be able to have a conversation with the AI")
        print("   3. See the session appear in active sessions")
    else:
        print("⚠️  Could not retrieve phone number for incoming call test")
    
    print()
    print("6️⃣ Manual Testing Instructions...")
    print()
    print("🔵 For Incoming Calls:")
    print("   1. Call the voice agent number")
    print("   2. Listen for greeting")
    print("   3. Start conversation with AI")
    print("   4. Check logs for language detection")
    print()
    print("🟡 For Outgoing Calls:")
    print("   1. Use the API endpoint or this test script")
    print("   2. Answer the call on your phone")
    print("   3. The AI should start talking to you")
    print("   4. Verify the call routes through GSM2 trunk")
    print()
    print("🔍 Troubleshooting:")
    print("   - Check voice agent logs for registration status")
    print("   - Verify Extension 200 is registered")
    print("   - Ensure GSM2 trunk is online")
    print("   - Check outgoing route uses GSM2 trunk (not voice-agent)")
    print()
    print("=" * 60)
    print("✅ Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
