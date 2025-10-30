"""
Test script to find the correct PBX recording URL
"""

import requests
import urllib3
from pbx_scraper import PBXScraper

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_recording_urls():
    """Test different recording URL formats to find the correct one"""
    
    # Initialize scraper
    scraper = PBXScraper()
    
    # Authenticate
    print("Authenticating...")
    if not scraper.authenticate():
        print("❌ Authentication failed")
        return
    
    print("✅ Authenticated\n")
    
    # Get a recording ID
    records = scraper.get_call_records()
    recording = None
    for record in records:
        if record['has_recording']:
            recording = record
            break
    
    if not recording:
        print("❌ No recordings found")
        return
    
    recording_id = recording['recording_id']
    print(f"Testing with recording ID: {recording_id}\n")
    
    # Try different URL formats
    test_urls = [
        f"/monitor/{recording_id}.wav",
        f"/monitor/{recording_id}",
        f"/public/report/play.php?file={recording_id}",
        f"/public/report/download.php?file={recording_id}",
        f"/public/recordings/{recording_id}.wav",
        f"/recordings/{recording_id}.wav",
        f"/recordings/{recording_id}",
        f"/var/spool/asterisk/monitor/{recording_id}.wav",
        f"/sounds/monitor/{recording_id}.wav",
        f"/media/{recording_id}.wav",
        # Without extension
        f"/public/report/play.php?id={recording_id}",
    ]
    
    print("Testing URLs...")
    print("=" * 80)
    
    working_urls = []
    
    for path in test_urls:
        full_url = f"{scraper.base_url}{path}"
        try:
            response = scraper.session.head(full_url, timeout=5, allow_redirects=True)
            status = response.status_code
            
            if status == 200:
                # Try to get content type
                content_type = response.headers.get('Content-Type', 'unknown')
                print(f"✅ {status} - {path}")
                print(f"   Content-Type: {content_type}")
                working_urls.append((path, content_type))
            elif status == 404:
                print(f"❌ {status} - {path}")
            else:
                print(f"⚠️  {status} - {path}")
        except Exception as e:
            print(f"❌ ERROR - {path}: {str(e)[:50]}")
    
    print("=" * 80)
    
    if working_urls:
        print(f"\n✅ Found {len(working_urls)} working URL(s):")
        for url, content_type in working_urls:
            print(f"   {url} ({content_type})")
        
        # Test actual download of the first working URL
        print(f"\n📥 Testing actual download from: {working_urls[0][0]}")
        test_url = f"{scraper.base_url}{working_urls[0][0]}"
        try:
            response = scraper.session.get(test_url, timeout=10)
            if response.status_code == 200:
                size_kb = len(response.content) / 1024
                print(f"   ✅ Downloaded {size_kb:.2f} KB")
                print(f"   Content-Type: {response.headers.get('Content-Type')}")
                
                # Save a sample
                with open("test_recording.wav", "wb") as f:
                    f.write(response.content)
                print(f"   💾 Saved to test_recording.wav")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("\n❌ No working URLs found")
        print("\n💡 Try checking the PBX web interface:")
        print("   1. Open https://192.168.50.50/public/report/index.php")
        print("   2. Click the Play button on a recording")
        print("   3. Check the Network tab in browser DevTools")
        print("   4. See what URL is actually requested")

if __name__ == "__main__":
    test_recording_urls()

