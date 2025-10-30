"""
PBX Call Recording Scraper
Scrapes call records and recordings from the PBX web interface
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional
import logging
import re
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class PBXScraper:
    """Scrapes call records and recordings from PBX web interface"""
    
    def __init__(self, base_url: str = "https://192.168.50.50", username: str = "admin", password: str = "admin"):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        # Disable SSL verification for local PBX (self-signed certificate)
        self.session.verify = False
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self._authenticated = False
    
    def authenticate(self) -> bool:
        """Authenticate with the PBX web interface"""
        try:
            # The PBX redirects to /public/ for login
            login_page_url = urljoin(self.base_url, "/public/")
            
            # First, get the login page to obtain CSRF token
            logger.info(f"Fetching login page from {login_page_url}")
            response = self.session.get(login_page_url)
            
            if response.status_code != 200:
                logger.error(f"Failed to get login page: {response.status_code}")
                return False
            
            # Parse the login form to get the postcode (CSRF token)
            soup = BeautifulSoup(response.text, 'html.parser')
            form = soup.find('form')
            
            if not form:
                logger.error("No login form found on page")
                return False
            
            # Extract the postcode token
            postcode_input = form.find('input', {'name': 'postcode'})
            postcode = postcode_input.get('value') if postcode_input else None
            
            if not postcode:
                logger.warning("No postcode token found, trying without it")
            
            # Submit login credentials with correct field names
            login_data = {
                'User': self.username,  # Capital U
                'Password': self.password,  # Capital P
            }
            
            if postcode:
                login_data['postcode'] = postcode
            
            logger.info(f"Submitting login form to {login_page_url}")
            response = self.session.post(login_page_url, data=login_data, allow_redirects=True)
            
            # Check if login was successful
            if response.status_code == 200:
                # Check if we're at the reports page and can see the table
                if 'blueTable' in response.text or 'table-sortable' in response.text:
                    self._authenticated = True
                    logger.info("✅ Successfully authenticated with PBX")
                    return True
                
                # If not at reports page, try to navigate there
                report_url = urljoin(self.base_url, "/public/report/index.php")
                logger.info(f"Navigating to reports page: {report_url}")
                response = self.session.get(report_url)
                
                if response.status_code == 200 and ('blueTable' in response.text or 'table-sortable' in response.text):
                    self._authenticated = True
                    logger.info("✅ Successfully authenticated with PBX")
                    return True
            
            logger.error(f"Authentication failed. Final URL: {response.url}")
            return False
            
        except Exception as e:
            logger.error(f"Error authenticating with PBX: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def get_call_records(self, page: int = 1) -> List[Dict]:
        """
        Fetch call records from the PBX
        
        Args:
            page: Page number to fetch (default: 1)
            
        Returns:
            List of call record dictionaries
        """
        if not self._authenticated:
            if not self.authenticate():
                raise Exception("Failed to authenticate with PBX")
        
        try:
            # Fetch the report page
            report_url = urljoin(self.base_url, "/public/report/index.php")
            params = {'page': page} if page > 1 else {}
            
            response = self.session.get(report_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch call records. Status: {response.status_code}")
                return []
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table
            table = soup.find('table', class_='blueTable')
            if not table:
                logger.warning("Could not find call records table")
                return []
            
            # Parse table rows
            tbody = table.find('tbody', id='rows')
            if not tbody:
                logger.warning("Could not find table body")
                return []
            
            records = []
            rows = tbody.find_all('tr', class_='edit_rows')
            
            for row in rows:
                try:
                    record = self._parse_row(row)
                    if record:
                        records.append(record)
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
            
            logger.info(f"Fetched {len(records)} call records from page {page}")
            return records
            
        except Exception as e:
            logger.error(f"Error fetching call records: {e}")
            return []
    
    def _parse_row(self, row) -> Optional[Dict]:
        """Parse a single table row into a call record dictionary"""
        try:
            cells = row.find_all('td')
            if len(cells) < 13:
                return None
            
            # Extract data from cells
            record_id = cells[0].get_text(strip=True)
            date_time = cells[1].get_text(strip=True)
            call_type_icon = cells[2].find('i')
            call_type = 'incoming' if 'glyphicon-import' in str(call_type_icon) else 'outgoing'
            src = cells[3].get_text(strip=True)
            dst = cells[4].get_text(strip=True)
            line = cells[5].get_text(strip=True)
            trunk = cells[6].get_text(strip=True)
            duration = cells[7].get_text(strip=True)
            billsec = cells[8].get_text(strip=True)
            last_app = cells[9].get_text(strip=True)
            status = cells[10].get_text(strip=True)
            
            # Check for recording
            record_cell = cells[11]
            recording_button = record_cell.find('button', onclick=True)
            recording_id = None
            has_recording = False
            
            if recording_button:
                onclick = recording_button.get('onclick', '')
                # Extract recording ID from Play('recording-id')
                match = re.search(r"Play\('([^']+)'\)", onclick)
                if match:
                    recording_id = match.group(1)
                    has_recording = True
            
            # Parse datetime (format: 30/10/2025 11:21:43)
            try:
                parsed_datetime = datetime.strptime(date_time, '%d/%m/%Y %H:%M:%S')
                iso_datetime = parsed_datetime.isoformat()
            except:
                iso_datetime = date_time
            
            return {
                'record_id': record_id,
                'datetime': iso_datetime,
                'date_display': date_time,
                'call_type': call_type,
                'src': src,
                'dst': dst,
                'line': line,
                'trunk': trunk,
                'duration': duration,
                'billsec': billsec,
                'last_app': last_app,
                'status': status,
                'has_recording': has_recording,
                'recording_id': recording_id,
                'recording_url': self._get_recording_url(recording_id) if recording_id else None
            }
            
        except Exception as e:
            logger.error(f"Error parsing row: {e}")
            return None
    
    def _get_recording_url(self, recording_id: str) -> str:
        """Get the URL to play/stream the recording"""
        # The PBX stores recordings as MP3 files in the rec/ directory
        # Format: /public/report/rec/{recording_id}.mp3
        return urljoin(self.base_url, f"/public/report/rec/{recording_id}.mp3")
    
    def get_recording_stream_url(self, recording_id: str) -> str:
        """
        Get the authenticated URL to stream a recording
        
        Args:
            recording_id: The recording ID from the PBX
            
        Returns:
            URL that can be used to stream the recording
        """
        return self._get_recording_url(recording_id)
    
    def search_by_phone_number(self, phone_number: str) -> List[Dict]:
        """
        Search for call records by phone number (src or dst)
        
        Args:
            phone_number: Phone number to search for
            
        Returns:
            List of matching call records
        """
        all_records = self.get_call_records()
        
        # Normalize phone number for comparison
        normalized = phone_number.replace('+', '').replace(' ', '').replace('-', '')
        
        matching = []
        for record in all_records:
            src_normalized = record['src'].replace('+', '').replace(' ', '').replace('-', '')
            dst_normalized = record['dst'].replace('+', '').replace(' ', '').replace('-', '')
            
            if normalized in src_normalized or normalized in dst_normalized:
                matching.append(record)
        
        return matching
    
    def get_recent_recordings(self, limit: int = 50) -> List[Dict]:
        """
        Get recent call records that have recordings
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of call records with recordings
        """
        all_records = self.get_call_records()
        
        # Filter to only records with recordings
        with_recordings = [r for r in all_records if r['has_recording']]
        
        # Return most recent first (already sorted by the table)
        return with_recordings[:limit]


# Singleton instance
_pbx_scraper = None

def get_pbx_scraper() -> PBXScraper:
    """Get or create the PBX scraper singleton instance"""
    global _pbx_scraper
    if _pbx_scraper is None:
        _pbx_scraper = PBXScraper()
    return _pbx_scraper

