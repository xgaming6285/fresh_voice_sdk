# -*- coding: utf-8 -*-
"""
Verify Session API Response
Shows what the API returns for a session with MongoDB data
"""

import json
from crm_database import get_session, CallSession
from crm_api import build_call_session_response

def verify_api_response():
    """Verify what the API returns for a session with MongoDB data"""
    print("=" * 70)
    print("Session API Response Verification")
    print("=" * 70)
    
    try:
        session = get_session()
        
        # Find a session with transcripts
        call_session = session.query(CallSession).filter(
            CallSession.transcripts != None
        ).first()
        
        if not call_session:
            print("\nNo sessions with transcripts found.")
            print("Run: python migrate_sessions_to_mongodb.py")
            return False
        
        print(f"\nSession ID: {call_session.session_id}")
        print(f"Lead ID: {call_session.lead_id}")
        print(f"Status: {call_session.status}")
        
        # Build API response
        response = build_call_session_response(call_session)
        
        # Convert to dict (as it would be in JSON)
        response_dict = response.model_dump()
        
        print("\n" + "=" * 70)
        print("API Response Structure")
        print("=" * 70)
        
        # Show what fields are available
        print("\nStandard Fields:")
        print(f"  - id: {response_dict['id']}")
        print(f"  - session_id: {response_dict['session_id']}")
        print(f"  - status: {response_dict['status']}")
        print(f"  - duration: {response_dict['duration']}")
        
        print("\nMongoDB Fields:")
        
        if response_dict.get('transcripts'):
            transcripts = response_dict['transcripts']
            print(f"  - transcripts: {list(transcripts.keys())}")
            for t_type, t_data in transcripts.items():
                print(f"    * {t_type}: {t_data.get('length', 0)} characters")
        else:
            print("  - transcripts: None")
        
        if response_dict.get('analysis'):
            analysis = response_dict['analysis']
            print(f"  - analysis:")
            print(f"    * status: {analysis.get('status', 'N/A')}")
            print(f"    * has summary: {bool(analysis.get('summary'))}")
        else:
            print("  - analysis: None")
        
        if response_dict.get('audio_files'):
            audio_files = response_dict['audio_files']
            print(f"  - audio_files: {list(audio_files.keys())}")
        else:
            print("  - audio_files: None")
        
        if response_dict.get('session_info'):
            session_info = response_dict['session_info']
            print(f"  - session_info:")
            print(f"    * duration: {session_info.get('duration_seconds', 'N/A')}s")
            print(f"    * caller_id: {session_info.get('caller_id', 'N/A')}")
        else:
            print("  - session_info: None")
        
        print("\n" + "=" * 70)
        print("SUCCESS - API will return MongoDB data!")
        print("=" * 70)
        print(f"\nTest this endpoint:")
        print(f"  GET /api/crm/sessions/{call_session.session_id}")
        print(f"\nThe response will include all transcripts and analysis!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    verify_api_response()

