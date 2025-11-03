"""Test phone number matching logic"""
import sys
sys.path.append('.')

# Test the matching logic
def test_phone_matching():
    """Test if phone number matching works correctly"""
    
    # Simulate CallSession data
    call_session_phone = "+359886068623"
    
    # Create phone_to_session mapping (simulating the endpoint logic)
    phone_to_session = {}
    
    # Original format
    phone_to_session[call_session_phone] = "SESSION_A"
    
    # Without + and with gate slot prefixes (9-19)
    clean_phone = call_session_phone.replace('+', '').replace(' ', '').replace('-', '')
    phone_to_session[clean_phone] = "SESSION_A"
    
    # With potential gate slot prefixes
    for slot in range(9, 20):
        prefixed = f"{slot}{clean_phone}"
        phone_to_session[prefixed] = "SESSION_A"
    
    # Just the last 9 digits (for partial matches)
    if len(clean_phone) >= 9:
        phone_to_session[clean_phone[-9:]] = "SESSION_A"
    
    print(f"Created {len(phone_to_session)} phone number variations")
    print(f"Sample variations:")
    for i, (phone, session) in enumerate(list(phone_to_session.items())[:10]):
        print(f"   {i+1}. {phone} -> {session}")
    print()
    
    # Test matching with PBX dst
    pbx_dst = "13359886068623"  # The actual format from PBX
    
    print(f"Testing PBX dst: {pbx_dst}")
    
    # Direct match
    if pbx_dst in phone_to_session:
        print(f"✅ Direct match found: {phone_to_session[pbx_dst]}")
        return True
    else:
        print("❌ No direct match")
        
        # Try matching without common prefixes and suffixes
        clean_dst = ''.join(filter(str.isdigit, pbx_dst))
        print(f"   Trying clean dst: {clean_dst}")
        
        if clean_dst in phone_to_session:
            print(f"   ✅ Clean match found: {phone_to_session[clean_dst]}")
            return True
        elif len(clean_dst) >= 9 and clean_dst[-9:] in phone_to_session:
            print(f"   ✅ Last 9 digits match found: {phone_to_session[clean_dst[-9:]]}")
            return True
        else:
            print("   ❌ No match found")
            return False

if __name__ == '__main__':
    test_phone_matching()

