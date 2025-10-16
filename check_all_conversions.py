# -*- coding: utf-8 -*-
"""
Check All Conversion Sources
Checks both MongoDB and local files for interested sessions
"""

import os
import json
from pathlib import Path
from pymongo import MongoClient

def check_all_conversions():
    """Check MongoDB and local files for conversions"""
    try:
        print("=" * 70)
        print("Check All Conversion Sources")
        print("=" * 70)
        
        # 1. Check MongoDB
        print("\n1. MONGODB:")
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        db = client['voice_agent_crm']
        client.admin.command('ping')
        
        mongo_interested = list(db.call_sessions.find({'analysis.status': 'interested'}))
        print(f"   Interested sessions: {len(mongo_interested)}")
        
        for session in mongo_interested:
            print(f"   - {session['session_id']}: {session.get('analysis', {}).get('summary', 'N/A')[:50]}")
        
        # 2. Check local session files
        print("\n2. LOCAL FILES (sessions/ folder):")
        sessions_dir = Path("sessions")
        
        if not sessions_dir.exists():
            print("   No sessions directory found")
            return
        
        local_interested = []
        
        for session_folder in sessions_dir.iterdir():
            if not session_folder.is_dir():
                continue
            
            analysis_file = session_folder / "analysis_result.json"
            if analysis_file.exists():
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    
                    if analysis.get('status') == 'interested':
                        local_interested.append({
                            'session_id': session_folder.name,
                            'analysis': analysis
                        })
                except Exception as e:
                    print(f"   Error reading {session_folder.name}: {e}")
        
        print(f"   Interested sessions in local files: {len(local_interested)}")
        
        for session in local_interested:
            print(f"   - {session['session_id']}: {session['analysis'].get('summary', 'N/A')[:50]}")
        
        # 3. Summary
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print("=" * 70)
        print(f"MongoDB interested sessions: {len(mongo_interested)}")
        print(f"Local file interested sessions: {len(local_interested)}")
        print(f"Total: {len(mongo_interested) + len(local_interested)}")
        
        if local_interested:
            print("\n" + "=" * 70)
            print("ACTION NEEDED:")
            print("=" * 70)
            print("Found interested sessions in LOCAL FILES!")
            print("The Conversions page reads from the API which checks summaries.")
            print("\nTo remove them:")
            print("1. Delete the analysis_result.json files from these session folders")
            print("2. Or run: python delete_local_conversions.py")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_all_conversions()

