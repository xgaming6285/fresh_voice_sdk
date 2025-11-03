import sys
sys.path.append('.')

from crm_database import get_session, CallSession, User, UserRole

def check_session():
    """Check specific session and agent55 access"""
    session = get_session()
    
    try:
        # Find agent55
        agent55 = session.query(User).filter(User.username == 'agent55').first()
        if not agent55:
            print("❌ agent55 not found!")
            return
        
        print(f"✅ Found agent55:")
        print(f"   ID: {agent55.id}")
        print(f"   Role: {agent55.role}")
        print(f"   Created by: {agent55.created_by_id}")
        print()
        
        # Check the specific session
        session_id = '8c0b2c8c-6012-47de-b32a-a3c5ea198399'
        call_session = session.query(CallSession).filter(CallSession.session_id == session_id).first()
        
        if not call_session:
            print(f"❌ Session {session_id} NOT FOUND in database!")
            print()
            print("Let's check all sessions for agent55:")
            agent55_sessions = session.query(CallSession).filter(CallSession.owner_id == agent55.id).all()
            print(f"   Agent55 has {len(agent55_sessions)} sessions:")
            for cs in agent55_sessions[:5]:  # Show first 5
                print(f"   - {cs.session_id} (owner: {cs.owner_id}, lead: {cs.lead_id})")
        else:
            print(f"✅ Session {session_id} FOUND in database:")
            print(f"   Owner ID: {call_session.owner_id}")
            print(f"   Lead ID: {call_session.lead_id}")
            print(f"   Status: {call_session.status}")
            print(f"   Start time: {call_session.start_time}")
            print()
            
            # Check ownership
            if call_session.owner_id == agent55.id:
                print("✅ This session BELONGS to agent55")
            else:
                print(f"❌ This session DOES NOT belong to agent55!")
                print(f"   It belongs to user_id: {call_session.owner_id}")
                
                # Find the actual owner
                actual_owner = session.query(User).filter(User.id == call_session.owner_id).first()
                if actual_owner:
                    print(f"   Owner username: {actual_owner.username}")
                    print(f"   Owner role: {actual_owner.role}")
        
    finally:
        session.close()

if __name__ == '__main__':
    check_session()

