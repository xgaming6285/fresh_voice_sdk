"""Test what the API actually returns for agent55"""
import sys
import json
sys.path.append('.')

from crm_database import get_session, CallSession, User, UserRole

def test_api_response_for_agent55():
    """Simulate the /api/crm/sessions endpoint for agent55"""
    db_session = get_session()
    
    try:
        # Find agent55
        agent55 = db_session.query(User).filter(User.username == 'agent55').first()
        print(f"Testing API response for: {agent55.username} (ID: {agent55.id}, Role: {agent55.role})")
        print()
        
        # This is the FIXED logic from crm_api.py for AGENT role (without .isnot(None))
        query = db_session.query(CallSession).filter(
            CallSession.owner_id == agent55.id
        )
        
        sessions = query.order_by(CallSession.started_at.desc()).limit(100).all()
        
        print(f"API would return {len(sessions)} sessions for agent55")
        print(f"First 10 session IDs:")
        for i, session in enumerate(sessions[:10]):
            print(f"   {i+1}. {session.session_id} (owner: {session.owner_id}, status: {session.status})")
        print()
        
        # Check if problematic session is in the list
        problem_id = '8c0b2c8c-6012-47de-b32a-a3c5ea198399'
        found = any(s.session_id == problem_id for s in sessions)
        
        if found:
            print(f"❌ ERROR: Problem session {problem_id} IS in the API response!")
        else:
            print(f"✅ CORRECT: Problem session {problem_id} is NOT in the API response")
        
        # Now test accessing that specific session
        print()
        print(f"Testing GET /api/crm/sessions/{problem_id} for agent55:")
        
        specific_session = db_session.query(CallSession).filter(
            CallSession.session_id == problem_id
        ).first()
        
        if specific_session:
            print(f"   Session found in database")
            print(f"   Owner ID: {specific_session.owner_id}")
            print(f"   Agent55 ID: {agent55.id}")
            
            if specific_session.owner_id == agent55.id:
                print(f"   ✅ Agent55 CAN access this session")
            else:
                print(f"   ❌ Agent55 CANNOT access (would get 403 Forbidden)")
        else:
            print(f"   Session not found in database (would get 404)")
        
    finally:
        db_session.close()

if __name__ == '__main__':
    test_api_response_for_agent55()

