"""
Debug script to inspect PBX login page structure
"""

import requests
from bs4 import BeautifulSoup
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def inspect_pbx_login():
    """Inspect the PBX login page to understand its structure"""
    
    base_url = "https://192.168.50.50"
    session = requests.Session()
    session.verify = False
    
    print("=" * 80)
    print("Inspecting PBX Login Page Structure")
    print("=" * 80)
    
    # Try accessing the report page directly
    print("\n1. Accessing report page directly...")
    try:
        response = session.get(f"{base_url}/public/report/index.php", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   URL: {response.url}")
        print(f"   Redirected: {response.history}")
        
        # Save the HTML to a file for inspection
        with open("pbx_page.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print("\n2. Analyzing page content...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for login form
        forms = soup.find_all('form')
        print(f"   Found {len(forms)} form(s)")
        
        for i, form in enumerate(forms, 1):
            print(f"\n   Form #{i}:")
            print(f"      Action: {form.get('action', 'N/A')}")
            print(f"      Method: {form.get('method', 'N/A')}")
            
            inputs = form.find_all('input')
            print(f"      Inputs ({len(inputs)}):")
            for inp in inputs:
                print(f"         - name='{inp.get('name')}', type='{inp.get('type')}', value='{inp.get('value')}'")
        
        # Check for table (if already logged in somehow)
        table = soup.find('table', class_='blueTable')
        if table:
            print("\n   ✅ Found blueTable - might be already accessible!")
        else:
            print("\n   ❌ No blueTable found - needs authentication")
        
        # Check for common authentication indicators
        print("\n3. Authentication indicators:")
        auth_indicators = {
            'login form': soup.find('form', {'action': lambda x: x and 'login' in x.lower()}) is not None,
            'username field': soup.find('input', {'name': lambda x: x and 'user' in x.lower()}) is not None,
            'password field': soup.find('input', {'type': 'password'}) is not None,
            'session/cookie': 'PHPSESSID' in response.cookies or 'session' in response.cookies.keys(),
        }
        
        for indicator, found in auth_indicators.items():
            status = "✅" if found else "❌"
            print(f"   {status} {indicator}: {found}")
        
        # Check cookies
        print("\n4. Cookies received:")
        for cookie in response.cookies:
            print(f"   - {cookie.name} = {cookie.value[:20]}...")
        
        # Look for any JavaScript that might handle authentication
        scripts = soup.find_all('script')
        print(f"\n5. Found {len(scripts)} script tags")
        
        # Check page title
        title = soup.find('title')
        print(f"\n6. Page title: {title.text if title else 'N/A'}")
        
        print("\n✅ HTML saved to pbx_page.html for detailed inspection")
        print("=" * 80)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_pbx_login()

