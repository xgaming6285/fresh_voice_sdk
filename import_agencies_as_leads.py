# -*- coding: utf-8 -*-
"""
Import agencies from agencies.json as leads into the CRM system
"""
import json
import requests
from typing import Dict, Any

# CRM API configuration
API_BASE_URL = "http://localhost:8000"
USERNAME = "agent4"
PASSWORD = "123123"

def login(username: str, password: str) -> str:
    """Login and get access token"""
    print(f"🔐 Logging in as {username}...")
    
    response = requests.post(
        f"{API_BASE_URL}/api/auth/login",
        json={"username": username, "password": password}
    )
    
    if response.status_code != 200:
        print(f"❌ Login failed: {response.text}")
        raise Exception(f"Login failed: {response.text}")
    
    data = response.json()
    access_token = data.get("access_token")
    print(f"✅ Login successful!")
    return access_token

def format_phone_number(phone: str) -> str:
    """
    Format phone number by removing leading 0 and adding +359
    Example: 0882817476 -> 882817476
    """
    if phone.startswith("0"):
        return phone[1:]  # Remove the leading 0
    return phone

def create_lead(agency: Dict[str, Any], access_token: str) -> bool:
    """Create a lead from agency data"""
    # Format phone number (remove leading 0)
    phone = format_phone_number(agency.get("agency_phone", ""))
    
    if not phone:
        print(f"⚠️  Skipping agency {agency.get('agency_name', 'Unknown')} - no phone number")
        return False
    
    # Build notes section with all agency info
    notes = f"""Agency Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agency Name: {agency.get('agency_name', 'N/A')}
Agency Address: {agency.get('agency_address', 'N/A')}
Broker Name: {agency.get('broker_name', 'N/A')}
Broker Office: {agency.get('broker_office', 'N/A')}
Broker Phone: {agency.get('broker_phone', 'N/A')}
Offer URL: {agency.get('offer_url', 'N/A')}
"""
    
    # Extract broker name for first_name/last_name
    broker_name = agency.get("broker_name", "")
    name_parts = broker_name.split(" ", 1) if broker_name else ["", ""]
    first_name = name_parts[0] if len(name_parts) > 0 else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""
    
    # Create lead payload
    lead_data = {
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "country_code": "+359",
        "country": "Bulgaria",
        "notes": notes.strip(),
        "gender": "unknown"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/crm/leads",
            json=lead_data,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            print(f"✅ Created lead: {broker_name} ({phone}) - {agency.get('agency_name', 'N/A')}")
            return True
        else:
            print(f"❌ Failed to create lead for {broker_name}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error creating lead for {broker_name}: {e}")
        return False

def main():
    """Main function to import agencies as leads"""
    print("=" * 60)
    print("📋 AGENCIES IMPORT SCRIPT")
    print("=" * 60)
    print()
    
    # Read agencies.json
    print("📂 Reading agencies.json...")
    try:
        with open("agencies.json", "r", encoding="utf-8") as f:
            agencies = json.load(f)
        print(f"✅ Found {len(agencies)} agencies in file")
    except FileNotFoundError:
        print("❌ Error: agencies.json not found!")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing agencies.json: {e}")
        return
    
    # Login
    try:
        access_token = login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    print()
    print("=" * 60)
    print(f"📞 Importing first 50 agencies as leads...")
    print("=" * 60)
    print()
    
    # Import first 50 agencies
    success_count = 0
    fail_count = 0
    
    for i, agency in enumerate(agencies[:50], 1):
        print(f"\n[{i}/50] Processing agency...")
        if create_lead(agency, access_token):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print()
    print("=" * 60)
    print("📊 IMPORT SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully imported: {success_count} leads")
    print(f"❌ Failed: {fail_count} leads")
    print(f"📋 Total processed: {success_count + fail_count} agencies")
    print()
    print("🎉 Import completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

