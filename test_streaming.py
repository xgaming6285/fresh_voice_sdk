"""
Test the backend streaming endpoint
"""
from pbx_scraper import get_pbx_scraper

pbx = get_pbx_scraper()
recording_id = "20251030-112143-200-10359988925337-1761816103.1597"

# Authenticate
pbx.authenticate()

# Get the recording URL
url = pbx.get_recording_stream_url(recording_id)
print(f"Recording URL: {url}")

# Try to fetch it
print("\nTesting download...")
response = pbx.session.get(url, stream=True)
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print(f"Content-Length: {response.headers.get('Content-Length')} bytes")

if response.status_code == 200:
    # Get first chunk
    chunk = next(response.iter_content(chunk_size=8192))
    print(f"First chunk size: {len(chunk)} bytes")
    print("✅ Recording streams successfully!")
else:
    print(f"❌ Failed to stream: {response.text}")

