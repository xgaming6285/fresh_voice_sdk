# -*- coding: utf-8 -*-
"""
Test API Fixes for MongoDB Migration
"""

import sys
from datetime import datetime

try:
    from crm_database import (
        init_database,
        get_session,
        User,
        UserRole,
        Lead,
        Campaign,
        CallSession,
        joinedload
    )
    
    print("=" * 60)
    print("API Fixes Test")
    print("=" * 60)
    
    # Test 1: Enum in filters
    print("\nTest 1: Testing enum in filters...")
    try:
        session = get_session()
        
        # This should convert the enum to string automatically
        agents = session.query(User).filter(User.role == UserRole.AGENT).all()
        print(f"OK - Found {len(agents)} agents (enum filter works)")
        
        session.close()
    except Exception as e:
        print(f"FAILED - Enum filter error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 2: joinedload compatibility
    print("\nTest 2: Testing joinedload compatibility...")
    try:
        session = get_session()
        
        # This should not raise an error
        query = session.query(CallSession).options(joinedload(CallSession.lead))
        sessions = query.all()
        print(f"OK - Found {len(sessions)} sessions (joinedload works)")
        
        session.close()
    except Exception as e:
        print(f"FAILED - joinedload error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Lead property access
    print("\nTest 3: Testing lead property access...")
    try:
        session = get_session()
        
        sessions = session.query(CallSession).all()
        if sessions:
            # Test accessing lead property
            first_session = sessions[0]
            lead = first_session.lead  # This should lazy-load the lead
            print(f"OK - Lead property access works (lead: {lead.full_name if lead else 'None'})")
        else:
            print("OK - No sessions to test (but property would work)")
        
        session.close()
    except Exception as e:
        print(f"FAILED - Lead property error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 4: Query with in_ operator
    print("\nTest 4: Testing in_ operator with list...")
    try:
        session = get_session()
        
        # Test in_ with a list
        user_ids = [1, 2, 3]
        users = session.query(User).filter(User.id.in_(user_ids)).all()
        print(f"OK - Found {len(users)} users with in_ operator")
        
        session.close()
    except Exception as e:
        print(f"FAILED - in_ operator error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All API fixes working correctly!")
    print("=" * 60)

except ImportError as e:
    print(f"FAILED - Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"FAILED - Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

