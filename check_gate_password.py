#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate VoIP Password Checker
This script helps you find the correct password for extension 200
"""

import json

def check_current_password():
    """Check the current password in config"""
    try:
        with open('asterisk_config.json', 'r') as f:
            config = json.load(f)
        
        print("🔍 Current Configuration:")
        print(f"   Username: {config['username']}")
        print(f"   Password: {config.get('password', 'NOT SET')}")
        print(f"   Gate IP: {config['host']}")
        
        return config
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def update_password():
    """Interactively update the password"""
    config = check_current_password()
    if not config:
        return
    
    print("\n" + "="*50)
    print("🔧 How to find the correct password:")
    print("="*50)
    print("1. Open your Gate VoIP web interface:")
    print(f"   http://{config['host']}")
    print("2. Login with admin credentials")
    print("3. Go to: PBX Settings > Internal Phones")
    print("4. Find extension 200 and check its password")
    print("5. The password should match what's in this config file")
    print()
    
    current_password = config.get('password', '')
    print(f"Current password in config: '{current_password}'")
    print()
    
    new_password = input("Enter the correct password from Gate VoIP (or press Enter to keep current): ").strip()
    
    if new_password and new_password != current_password:
        config['password'] = new_password
        
        try:
            with open('asterisk_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Password updated to: '{new_password}'")
            print("💡 Now try running the voice agent again")
        except Exception as e:
            print(f"❌ Error updating config: {e}")
    else:
        print("⏭️ Password unchanged")

def main():
    print("Gate VoIP Extension 200 Password Checker")
    print("="*50)
    print()
    
    print("📋 Common Gate VoIP Extension 200 Passwords:")
    print("   - 123123 (default)")
    print("   - 200 (extension number)")
    print("   - admin")
    print("   - password")
    print("   - (empty/blank)")
    print()
    
    check_current_password()
    print()
    
    choice = input("Do you want to update the password? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        update_password()
    else:
        print("💡 Check your Gate VoIP web interface to verify the extension 200 password")

if __name__ == "__main__":
    main()
