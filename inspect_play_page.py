from pbx_scraper import PBXScraper
from bs4 import BeautifulSoup

scraper = PBXScraper()
scraper.authenticate()

recording_id = "20251030-112143-200-10359988925337-1761816103.1597"

# Get the play.php page
url = f"{scraper.base_url}/public/report/play.php?file={recording_id}"
response = scraper.session.get(url)

print("Response from play.php:")
print("=" * 80)
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print(f"Content-Length: {len(response.content)} bytes")
print("\nHTML Content:")
print("=" * 80)
print(response.text[:2000])
print("\n" + "=" * 80)

# Parse HTML to find audio source
soup = BeautifulSoup(response.text, 'html.parser')

# Look for audio/source tags
audio_tags = soup.find_all('audio')
source_tags = soup.find_all('source')
links = soup.find_all('a', href=True)

print("\nLooking for audio elements...")
if audio_tags:
    for audio in audio_tags:
        print(f"Audio tag: {audio}")
        
if source_tags:
    for source in source_tags:
        print(f"Source tag: {source}")

# Look for any URLs that might be the audio file
print("\nLooking for file URLs...")
for link in links:
    href = link.get('href')
    if any(ext in href.lower() for ext in ['.wav', '.mp3', 'monitor', 'recording', 'download']):
        print(f"Found link: {href}")

# Look for JavaScript that might load the audio
scripts = soup.find_all('script')
for script in scripts:
    if script.string and ('wav' in script.string.lower() or 'audio' in script.string.lower() or 'src' in script.string.lower()):
        print(f"\nScript with audio reference:")
        print(script.string[:500])

