from pbx_scraper import PBXScraper
from bs4 import BeautifulSoup

scraper = PBXScraper()
scraper.authenticate()

# Get the reports page
response = scraper.session.get(f"{scraper.base_url}/public/report/index.php")

soup = BeautifulSoup(response.text, 'html.parser')

# Find all script tags
scripts = soup.find_all('script')

print("Looking for Play function in reports page...")
print("=" * 80)

for i, script in enumerate(scripts):
    if script.string and ('Play' in script.string or 'play' in script.string):
        print(f"\n=== Script {i+1} ===")
        print(script.string)
        print("\n")

