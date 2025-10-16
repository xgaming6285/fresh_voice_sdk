# -*- coding: utf-8 -*-
"""
View Conversion Records
Shows all sessions marked as 'interested' (conversions)
"""

from pymongo import MongoClient
from datetime import datetime

def view_conversions():
    """View all conversion records"""
    try:
        print("=" * 70)
        print("Conversion Records (Interested Sessions)")
        print("=" * 70)
        
        # Connect to MongoDB
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        db = client['voice_agent_crm']
        
        # Test connection
        client.admin.command('ping')
        print(f"\nConnected to MongoDB: {db.name}")
        
        # Get all sessions with stats
        total_sessions = db.call_sessions.count_documents({})
        interested = db.call_sessions.count_documents({'analysis.status': 'interested'})
        not_interested = db.call_sessions.count_documents({'analysis.status': 'not_interested'})
        no_analysis = db.call_sessions.count_documents({'analysis': None})
        
        print(f"\nSession Statistics:")
        print(f"  - Total sessions: {total_sessions}")
        print(f"  - Interested (conversions): {interested}")
        print(f"  - Not interested: {not_interested}")
        print(f"  - No analysis: {no_analysis}")
        
        if interested > 0:
            print(f"\n" + "=" * 70)
            print("Interested Sessions (Conversions):")
            print("=" * 70)
            
            sessions = db.call_sessions.find({'analysis.status': 'interested'}).sort('started_at', -1)
            
            for i, session in enumerate(sessions, 1):
                print(f"\n{i}. Session ID: {session.get('session_id', 'N/A')}")
                print(f"   Lead ID: {session.get('lead_id', 'N/A')}")
                print(f"   Called: {session.get('called_number', 'N/A')}")
                print(f"   Status: {session.get('status', 'N/A')}")
                print(f"   Started: {session.get('started_at', 'N/A')}")
                if session.get('analysis'):
                    print(f"   Summary: {session['analysis'].get('summary', 'N/A')[:100]}...")
        
        print("\n" + "=" * 70)
        print("To delete conversion records, run: python delete_conversions.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    view_conversions()

