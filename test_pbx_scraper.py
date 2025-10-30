"""
Test script for PBX Scraper
Run this to test the PBX scraping functionality
"""

from pbx_scraper import PBXScraper
import json

def test_pbx_scraper():
    """Test PBX scraper functionality"""
    
    print("=" * 60)
    print("Testing PBX Scraper")
    print("=" * 60)
    
    # Initialize scraper
    scraper = PBXScraper(
        base_url="https://192.168.50.50",
        username="admin",
        password="admin"
    )
    
    # Test authentication
    print("\n1. Testing authentication...")
    if scraper.authenticate():
        print("   ✅ Authentication successful")
    else:
        print("   ❌ Authentication failed")
        return
    
    # Test fetching call records
    print("\n2. Fetching call records...")
    records = scraper.get_call_records()
    print(f"   Found {len(records)} call records")
    
    # Display some sample records
    if records:
        print("\n3. Sample call records:")
        for i, record in enumerate(records[:5], 1):  # Show first 5
            print(f"\n   Record #{i}:")
            print(f"      Date: {record['date_display']}")
            print(f"      Type: {record['call_type']}")
            print(f"      From: {record['src']}")
            print(f"      To: {record['dst']}")
            print(f"      Duration: {record['duration']}")
            print(f"      Status: {record['status']}")
            print(f"      Has Recording: {record['has_recording']}")
            if record['has_recording']:
                print(f"      Recording ID: {record['recording_id']}")
                print(f"      Recording URL: {record['recording_url']}")
    
    # Test getting recordings only
    print("\n4. Fetching records with recordings...")
    recordings = scraper.get_recent_recordings(limit=10)
    print(f"   Found {len(recordings)} records with recordings")
    
    if recordings:
        print("\n5. Recordings:")
        for i, rec in enumerate(recordings, 1):
            print(f"\n   Recording #{i}:")
            print(f"      Date: {rec['date_display']}")
            print(f"      From: {rec['src']} -> To: {rec['dst']}")
            print(f"      Recording ID: {rec['recording_id']}")
            print(f"      URL: {rec['recording_url']}")
    
    # Test search by phone number
    print("\n6. Testing search by phone number...")
    test_number = "+359988925337"
    matches = scraper.search_by_phone_number(test_number)
    print(f"   Found {len(matches)} matches for {test_number}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_pbx_scraper()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

