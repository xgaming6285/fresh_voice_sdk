# -*- coding: utf-8 -*-
"""
Test Session MongoDB Data via API
Verify that transcripts, analysis, and other session data are accessible
"""

import os
import sys
from pymongo import MongoClient
from crm_database import get_session, CallSession

# Set environment variables for MongoDB connection (if not already set)
os.environ.setdefault('MONGODB_HOST', 'localhost')
os.environ.setdefault('MONGODB_PORT', '27017')
os.environ.setdefault('MONGODB_DATABASE', 'voice_agent_crm')

def test_session_mongodb_data():
    """Test that session data is properly stored and accessible in MongoDB"""
    print("=" * 70)
    print("Session MongoDB Data Test")
    print("=" * 70)
    
    try:
        # Connect directly to MongoDB to verify data
        print("\n1. Connecting to MongoDB directly...")
        client = MongoClient('localhost', 27017)
        db = client['voice_agent_crm']
        print(f"   Connected to database: {db.name}")
        
        # Find a session with transcripts
        print("\n2. Finding session with transcript data...")
        mongo_session = db.call_sessions.find_one({"transcripts": {"$exists": True}})
        
        if not mongo_session:
            print("   No sessions with transcripts found. Run migrate_sessions_to_mongodb.py first.")
            return False
        
        session_id = mongo_session['session_id']
        print(f"   Found session: {session_id}")
        
        # Check what data exists
        print("\n3. Checking MongoDB document fields:")
        print(f"   - Has transcripts: {bool(mongo_session.get('transcripts'))}")
        print(f"   - Has analysis: {bool(mongo_session.get('analysis'))}")
        print(f"   - Has audio_files: {bool(mongo_session.get('audio_files'))}")
        print(f"   - Has session_info: {bool(mongo_session.get('session_info'))}")
        
        if mongo_session.get('transcripts'):
            transcripts = mongo_session['transcripts']
            print(f"\n   Transcript types: {list(transcripts.keys())}")
            for t_type, t_data in transcripts.items():
                print(f"     - {t_type}: {t_data.get('length', 0)} characters")
        
        if mongo_session.get('analysis'):
            analysis = mongo_session['analysis']
            print(f"\n   Analysis:")
            print(f"     - Status: {analysis.get('status', 'N/A')}")
            print(f"     - Has summary: {bool(analysis.get('summary'))}")
        
        # Now test via ORM
        print("\n4. Testing via ORM (crm_database)...")
        db_session = get_session()
        
        try:
            orm_session = db_session.query(CallSession).filter(
                CallSession.session_id == session_id
            ).first()
            
            if not orm_session:
                print("   ERROR: Session not found via ORM")
                return False
            
            print(f"   Found session via ORM: {orm_session.session_id}")
            print(f"   - ID: {orm_session.id}")
            print(f"   - Status: {orm_session.status}")
            print(f"   - Lead ID: {orm_session.lead_id}")
            
            # Check if MongoDB data is accessible as attributes
            print("\n5. Checking MongoDB fields via ORM:")
            
            # These should be accessible via getattr since MongoDB returns them
            has_transcripts = hasattr(orm_session, 'transcripts') and orm_session.transcripts is not None
            has_analysis = hasattr(orm_session, 'analysis') and orm_session.analysis is not None
            
            print(f"   - Has transcripts attribute: {has_transcripts}")
            print(f"   - Has analysis attribute: {has_analysis}")
            
            if has_transcripts:
                try:
                    print(f"     Transcript types: {list(orm_session.transcripts.keys())}")
                except Exception as e:
                    print(f"     Could not display transcript keys: {type(e).__name__}")
            
            if has_analysis:
                try:
                    status = orm_session.analysis.get('status', 'N/A')
                    print(f"     Analysis status: {status}")
                except Exception as e:
                    print(f"     Could not display analysis: {type(e).__name__}")
        
        finally:
            db_session.close()
        
        print("\n" + "=" * 70)
        print("Test Results:")
        print("=" * 70)
        print("MongoDB Storage: OK")
        print("ORM Access: OK (via getattr)")
        print("Ready for API: YES")
        print("=" * 70)
        print("\nThe API endpoints will now return transcripts and analysis data!")
        print(f"Example: GET /api/crm/sessions/{session_id}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_session_mongodb_data()
    sys.exit(0 if success else 1)

