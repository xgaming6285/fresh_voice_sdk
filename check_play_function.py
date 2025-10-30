from bs4 import BeautifulSoup

with open('pbx_page.html', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

# Find all script tags
scripts = soup.find_all('script')

print("Looking for Play function...")
print("=" * 80)

for i, script in enumerate(scripts):
    if script.string and 'Play' in script.string:
        print(f"\n=== Script {i+1} ===")
        print(script.string[:1000])
        print("...")

