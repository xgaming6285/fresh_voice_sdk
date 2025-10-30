"""
Import Real Estate Agencies from JSON file to CRM
"""
import json
import requests
import sys
import time
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
AGENCIES_FILE = r"C:\Users\EVLSV69\Desktop\Г-н Ройс\agencies.json"

# Default admin credentials (replace with your actual credentials)
USERNAME = "admin2"
PASSWORD = "123123"

def load_agencies():
    """Load agencies from JSON file"""
    try:
        with open(AGENCIES_FILE, 'r', encoding='utf-8') as f:
            agencies = json.load(f)
        print(f"[OK] Loaded {len(agencies)} agencies from file")
        return agencies
    except FileNotFoundError:
        print(f"[ERROR] File not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON")
        sys.exit(1)

def convert_phone(phone):
    """Remove leading 0 from Bulgarian phone numbers"""
    if phone.startswith('0'):
        return phone[1:]  # Remove the leading 0
    return phone

def wait_for_server():
    """Wait for server to start"""
    print("[INFO] Waiting for CRM server to start...")
    for i in range(30):  # Try for 30 seconds
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("[OK] Server is ready!")
                return True
        except:
            time.sleep(1)
            print(".", end="", flush=True)
    print("\n[ERROR] Server did not start in time")
    return False

def login_and_get_token():
    """Login to CRM and get JWT token"""
    try:
        print(f"[INFO] Logging in as {USERNAME}...")
        response = requests.post(
            f"{API_BASE_URL}/api/auth/login",
            json={"username": USERNAME, "password": PASSWORD}
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            user = data.get('user', {})
            print(f"[OK] Logged in as: {user.get('full_name', USERNAME)}")
            return token
        else:
            print(f"[ERROR] Login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Login error: {e}")
        return None

def delete_existing_leads(session, token, agencies):
    """Delete existing leads with matching phone numbers"""
    print("\n[INFO] Checking for existing leads to delete...")
    deleted_count = 0
    
    # Get all possible phone formats to check
    phone_variants = []
    for agency in agencies:
        phone = agency['phone']
        # Create all possible variants
        phone_variants.append(phone)  # Original: 0882817451
        phone_variants.append(phone[1:] if phone.startswith('0') else phone)  # Without 0: 882817451
        phone_variants.append('+359' + phone[1:] if phone.startswith('0') else '+359' + phone)  # Full: +359882817451
    
    try:
        # Get all leads
        response = session.get(
            f"{API_BASE_URL}/api/crm/leads?per_page=1000",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            leads = data.get('leads', [])
            
            # Delete all matching leads
            for lead in leads:
                lead_phone = lead.get('phone', '')
                if lead_phone in phone_variants:
                    delete_response = session.delete(
                        f"{API_BASE_URL}/api/crm/leads/{lead['id']}",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    if delete_response.status_code == 200:
                        deleted_count += 1
                        print(f"[DELETE] Removed: {lead_phone}")
    except Exception as e:
        print(f"[ERROR] During deletion: {e}")
    
    print(f"[OK] Deleted {deleted_count} existing leads\n")
    return deleted_count

def import_lead(agency, session, token):
    """Import a single lead into CRM"""
    # Convert phone number
    phone = convert_phone(agency['phone'])
    
    # Split agency name into first/last name (or use full name as first name)
    full_name = agency['agency_name']
    
    lead_data = {
        "first_name": full_name,
        "last_name": "",  # Empty for agencies
        "phone": phone,  # Phone without country code (e.g., 882817451)
        "country": "Bulgaria",
        "country_code": "+359",  # Country code as +359
        "gender": "unknown",
        "notes": f"Real Estate Agency: {full_name}"
    }
    
    try:
        response = session.post(
            f"{API_BASE_URL}/api/crm/leads",
            json=lead_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            # Avoid printing Cyrillic characters
            print(f"[OK] Added: {phone}")
            return True
        else:
            print(f"[FAIL] {phone} - {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] {phone}")
        return False

def main():
    print("=" * 60)
    print("Real Estate Agency Lead Import Tool")
    print("=" * 60)
    
    # Load agencies
    agencies = load_agencies()
    
    # Wait for server
    if not wait_for_server():
        sys.exit(1)
    
    # Login and get token
    token = login_and_get_token()
    if not token:
        print("[ERROR] Failed to authenticate. Cannot continue.")
        sys.exit(1)
    
    # Create session for API calls
    session = requests.Session()
    
    # First, delete existing leads using admin2 token (they own the current leads)
    print("\n[INFO] Deleting existing leads from admin2 account...")
    delete_existing_leads(session, token, agencies)
    
    print(f"Starting import of {len(agencies)} agencies as {USERNAME}...\n")
    
    success_count = 0
    fail_count = 0
    
    for i, agency in enumerate(agencies, 1):
        print(f"[{i}/{len(agencies)}] ", end="")
        if import_lead(agency, session, token):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Imported: {success_count}")
    print(f"[FAILED] Failed: {fail_count}")
    print(f"[TOTAL] Total: {len(agencies)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

