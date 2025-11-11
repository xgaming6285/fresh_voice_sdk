#!/usr/bin/env python3
"""
Simple test script to make an outbound call and observe SIP response codes (100, 180, 200)
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
PHONE_NUMBER = "+359988925337"
GATE_SLOT = 9  # Using slot 9 as requested

def make_test_call():
    """Make a test outbound call to observe SIP response codes"""
    
    print("=" * 70)
    print("🧪 TEST OUTBOUND CALL - SIP Response Code Monitor")
    print("=" * 70)
    print(f"📞 Phone Number: {PHONE_NUMBER}")
    print(f"📍 Gate Slot: {GATE_SLOT}")
    print(f"🌐 API: {API_BASE_URL}")
    print("=" * 70)
    
    # First, generate a greeting
    print("\n🎤 Step 1: Generating greeting...")
    greeting_payload = {
        "phone_number": PHONE_NUMBER,
        "language": "Bulgarian",
        "company_name": "Test Company",
        "caller_name": "Test Agent",
        "product_name": "Test Product",
        "objective": "testing SIP codes"
    }
    
    try:
        greeting_response = requests.post(
            f"{API_BASE_URL}/api/generate_greeting",
            json=greeting_payload,
            timeout=60
        )
        
        if greeting_response.status_code == 200:
            greeting_data = greeting_response.json()
            print(f"✅ Greeting generated: {greeting_data.get('greeting_file', 'N/A')}")
            greeting_file = greeting_data.get('greeting_file')
        else:
            print(f"⚠️  Greeting generation failed: {greeting_response.status_code}")
            print("Continuing with default greeting...")
            greeting_file = None
            
    except Exception as e:
        print(f"❌ Error generating greeting: {e}")
        print("Continuing with default greeting...")
        greeting_file = None
    
    # Now make the call
    print(f"\n📞 Step 2: Making outbound call...")
    print(f"🎯 Watch the main application logs for:")
    print(f"   ✅ '100 Trying (Call Initiated)'")
    print(f"   🔔 '180 Ringing (Phone Ringing)' ← Looking for this!")
    print(f"   ✅ '200 OK (Success Response)'")
    print("-" * 70)
    
    call_payload = {
        "phone_number": PHONE_NUMBER,
        "gate_slot": GATE_SLOT,
        "custom_config": {
            "company": "Test Company",
            "caller": "Test Agent",
            "product": "Test Product - SIP Code Testing",
            "objective": "test_call",
            "urgency": "low",
            "benefits": ["testing SIP response codes", "verifying 180 ringing detection"],
            "offers": "This is a test call to observe SIP signaling.",
            "objection_strategy": "understanding"
        }
    }
    
    # Add greeting file if we generated one
    if greeting_file:
        call_payload["greeting_file"] = greeting_file
    
    try:
        print(f"\n🚀 Sending call request...")
        call_response = requests.post(
            f"{API_BASE_URL}/api/test/make_call",  # Using test endpoint (no auth required)
            json=call_payload,
            timeout=10
        )
        
        if call_response.status_code == 200:
            result = call_response.json()
            print(f"\n✅ Call initiated successfully!")
            print(f"📋 Session ID: {result.get('session_id', 'N/A')}")
            print(f"📞 Status: {result.get('status', 'N/A')}")
            
            if 'message' in result:
                print(f"💬 Message: {result['message']}")
            
            print("\n" + "=" * 70)
            print("👀 CHECK THE MAIN APPLICATION LOGS ABOVE")
            print("   Look for the SIP response sequence:")
            print("   1. 📞 Received SIP 100 Trying (Call Initiated)")
            print("   2. 📞 Call initiated - processing call to 9359988925337")
            print("   3. 🔗 Asterisk Linked ID (Call-Wide): <linkedid>  ← NEW!")
            print("   4. 🔔 Received SIP 180 Ringing (Phone Ringing)  ← TARGET!")
            print("   5. 🔔 Phone ringing - outbound call to 9359988925337")
            print("   6. 📞 Received SIP 200 OK (Success Response)")
            print("   7. ✅ Outbound call to 9359988925337 answered!")
            print("=" * 70)
            
            print("\n💡 Note: If you see 100 Trying immediately followed by 200 OK,")
            print("   it means the phone answered too quickly (no ringing phase).")
            print("   Try calling a number that will ring for a few seconds.")
            print("\n🔗 Linked ID: This is the call-wide unique ID from Asterisk.")
            print("   All legs of the call share this ID (useful for CDR tracking).")
            print("   To enable it, configure your Asterisk dialplan to send")
            print("   X-Asterisk-Linkedid header (see asterisk_dialplan.conf).")
            
            return True
            
        elif call_response.status_code == 503:
            print(f"\n❌ Service temporarily unavailable")
            print(f"💡 Make sure Extension 200 is registered (check startup logs)")
            return False
            
        elif call_response.status_code == 403:
            print(f"\n❌ Authentication/Authorization failed")
            error = call_response.json()
            print(f"💡 {error.get('detail', 'Unknown error')}")
            return False
            
        else:
            print(f"\n❌ Call failed with status code: {call_response.status_code}")
            try:
                error = call_response.json()
                print(f"Error details: {json.dumps(error, indent=2)}")
            except:
                print(f"Response: {call_response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection error!")
        print(f"💡 Make sure the voice agent is running on {API_BASE_URL}")
        return False
        
    except requests.exceptions.Timeout:
        print(f"\n⏰ Request timeout!")
        print(f"💡 Call may still be processing - check main application logs")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Voice Agent is online")
            print(f"📊 Status: {health.get('status', 'N/A')}")
            if 'extension_registered' in health:
                if health['extension_registered']:
                    print(f"✅ Extension 200 is registered - ready for outbound calls")
                else:
                    print(f"⚠️  Extension 200 is NOT registered - outbound calls will fail")
                    print(f"💡 Wait for extension registration to complete")
            return True
        else:
            print(f"⚠️  API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {API_BASE_URL}")
        print(f"💡 Make sure the voice agent is running")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Checking voice agent status...")
    if not check_health():
        print("\n❌ Voice agent is not available. Please start it first:")
        print("   python windows_voice_agent.py")
        exit(1)
    
    print("\n" + "=" * 70)
    input("Press ENTER to initiate the test call...")
    print()
    
    success = make_test_call()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Test completed - check logs above for SIP response codes")
    else:
        print("❌ Test failed - check error messages above")
    print("=" * 70)

