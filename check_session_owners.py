import sys
sys.path.append('.')

from crm_database import get_session, CallSession, User, UserRole

def check_owners():
    """Check session ownership and who can see what"""
    session = get_session()
    
    try:
        # Find agent55
        agent55 = session.query(User).filter(User.username == 'agent55').first()
        print(f"✅ agent55:")
        print(f"   ID: {agent55.id}")
        print(f"   Role: {agent55.role}")
        print(f"   Created by: {agent55.created_by_id}")
        print()
        
        # Check who owns the session
        session_id = '8c0b2c8c-6012-47de-b32a-a3c5ea198399'
        call_session = session.query(CallSession).filter(CallSession.session_id == session_id).first()
        
        print(f"Session {session_id}:")
        print(f"   Owner ID: {call_session.owner_id}")
        
        actual_owner = session.query(User).filter(User.id == call_session.owner_id).first()
        if actual_owner:
            print(f"   Owner username: {actual_owner.username}")
            print(f"   Owner role: {actual_owner.role}")
            print(f"   Owner created by: {actual_owner.created_by_id}")
        print()
        
        # Check all sessions in the database and their owners
        all_sessions = session.query(CallSession).all()
        print(f"Total sessions in database: {len(all_sessions)}")
        print()
        
        # Group by owner
        from collections import defaultdict
        sessions_by_owner = defaultdict(list)
        for cs in all_sessions:
            sessions_by_owner[cs.owner_id].append(cs.session_id)
        
        print("Sessions by owner:")
        for owner_id, session_ids in sessions_by_owner.items():
            owner = session.query(User).filter(User.id == owner_id).first()
            username = owner.username if owner else "Unknown"
            print(f"   User {owner_id} ({username}): {len(session_ids)} sessions")
        print()
        
        # Check agent55's sessions
        agent55_sessions = session.query(CallSession).filter(CallSession.owner_id == agent55.id).all()
        print(f"Agent55 has {len(agent55_sessions)} sessions:")
        for cs in agent55_sessions[:10]:  # Show first 10
            print(f"   - {cs.session_id}")
        
    finally:
        session.close()

if __name__ == '__main__':
    check_owners()

