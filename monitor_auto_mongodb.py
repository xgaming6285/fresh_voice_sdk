# -*- coding: utf-8 -*-
"""
Monitor Automatic MongoDB Integration
Watch for new sessions and their data being saved to MongoDB in real-time
"""

from pymongo import MongoClient
import time
from datetime import datetime

def monitor_sessions():
    """Monitor MongoDB for new session data"""
    try:
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        db = client['voice_agent_crm']
        
        # Test connection
        client.admin.command('ping')
        
        print("=" * 70)
        print("AUTOMATIC MONGODB INTEGRATION MONITOR")
        print("=" * 70)
        print(f"\nConnected to MongoDB: {db.name}")
        print("\nMonitoring for new sessions and their data...")
        print("Make a call now and watch the data appear automatically!\n")
        print("-" * 70)
        
        last_count = db.call_sessions.count_documents({})
        last_check = {}  # Track what data each session has
        
        while True:
            current_count = db.call_sessions.count_documents({})
            
            # Check for new sessions
            if current_count > last_count:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] NEW SESSION detected! Total: {current_count}")
                
                latest = db.call_sessions.find_one(sort=[('created_at', -1)])
                session_id = latest['session_id']
                
                print(f"  Session ID: {session_id}")
                print(f"  Lead ID: {latest.get('lead_id', 'N/A')}")
                print(f"  Status: {latest.get('status', 'N/A')}")
                
                last_check[session_id] = {
                    'session_info': bool(latest.get('session_info')),
                    'transcripts': bool(latest.get('transcripts')),
                    'analysis': bool(latest.get('analysis'))
                }
                
                last_count = current_count
            
            # Check for updates to existing sessions
            recent_sessions = db.call_sessions.find().sort('updated_at', -1).limit(5)
            
            for session in recent_sessions:
                session_id = session['session_id']
                
                current_state = {
                    'session_info': bool(session.get('session_info')),
                    'transcripts': bool(session.get('transcripts')),
                    'analysis': bool(session.get('analysis'))
                }
                
                # Check if this is the first time we're seeing this session
                if session_id not in last_check:
                    last_check[session_id] = current_state
                    continue
                
                # Check for changes
                previous_state = last_check[session_id]
                
                if current_state != previous_state:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] UPDATE for session {session_id[:8]}...")
                    
                    if current_state['session_info'] != previous_state['session_info']:
                        print("  [OK] Session info saved to MongoDB")
                    
                    if current_state['transcripts'] != previous_state['transcripts']:
                        print("  [OK] Transcripts saved to MongoDB")
                        if session.get('transcripts'):
                            transcript_types = list(session['transcripts'].keys())
                            print(f"       Types: {', '.join(transcript_types)}")
                    
                    if current_state['analysis'] != previous_state['analysis']:
                        print("  [OK] Analysis saved to MongoDB")
                        if session.get('analysis'):
                            status = session['analysis'].get('status', 'N/A')
                            print(f"       Status: {status}")
                    
                    last_check[session_id] = current_state
            
            time.sleep(3)  # Check every 3 seconds
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure MongoDB is running on localhost:27017")
        return False

if __name__ == "__main__":
    monitor_sessions()

