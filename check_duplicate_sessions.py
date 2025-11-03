import sys
sys.path.append('.')

from crm_database import get_session, CallSession, User

def check_duplicate_session_ids():
    """Check for duplicate session IDs"""
    session = get_session()
    
    try:
        # Check the specific session
        problem_session_id = '8c0b2c8c-6012-47de-b32a-a3c5ea198399'
        
        all_instances = session.query(CallSession).filter(
            CallSession.session_id == problem_session_id
        ).all()
        
        print(f"Found {len(all_instances)} instances of session {problem_session_id}:")
        for instance in all_instances:
            owner = session.query(User).filter(User.id == instance.owner_id).first()
            owner_name = owner.username if owner else "Unknown"
            print(f"   - ID: {instance.id}, Owner: {owner_name} (user_id: {instance.owner_id}), Status: {instance.status}")
        print()
        
        # Check for any duplicate session_ids in the database
        from sqlalchemy import func
        duplicates = session.query(
            CallSession.session_id,
            func.count(CallSession.id).label('count')
        ).group_by(CallSession.session_id).having(func.count(CallSession.id) > 1).all()
        
        print(f"Total duplicate session_ids in database: {len(duplicates)}")
        if len(duplicates) > 0:
            print(f"Showing first 10 duplicates:")
            for dup_session_id, count in duplicates[:10]:
                print(f"   Session {dup_session_id}: {count} instances")
        
    finally:
        session.close()

if __name__ == '__main__':
    check_duplicate_session_ids()

