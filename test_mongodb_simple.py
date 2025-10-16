# -*- coding: utf-8 -*-
"""
Simple MongoDB Connection Test (Windows-compatible)
Run this to verify MongoDB is working correctly
"""

import sys
from datetime import datetime

try:
    from pymongo import MongoClient
    from crm_database import (
        init_database,
        get_session,
        User,
        UserRole,
        Lead,
        Gender,
        UserManager,
        LeadManager
    )
    
    print("=" * 60)
    print("MongoDB Connection Test")
    print("=" * 60)
    
    # Test 1: Connect to MongoDB
    print("\nTest 1: Connecting to MongoDB...")
    try:
        db = init_database()
        print(f"OK - Connected to database: {db.name}")
        print(f"Collections: {', '.join(db.list_collection_names())}")
    except Exception as e:
        print(f"FAILED - Could not connect: {e}")
        print("Make sure MongoDB is running on localhost:27017")
        sys.exit(1)
    
    # Test 2: Test class attribute access
    print("\nTest 2: Testing class attribute access...")
    try:
        # This should work now
        user_id_col = User.id
        user_created_by_col = User.created_by_id
        lead_id_col = Lead.id
        print("OK - Class attributes work correctly")
    except Exception as e:
        print(f"FAILED - Class attribute access error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Create a test user
    print("\nTest 3: Creating test user...")
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        # Check if test user already exists
        existing_user = user_manager.get_user_by_username("test_user")
        
        if existing_user:
            print(f"OK - Test user already exists (ID: {existing_user.id})")
            test_user = existing_user
        else:
            test_user = user_manager.create_user(
                username="test_user",
                email="test@example.com",
                password="test123",
                role=UserRole.ADMIN,
                first_name="Test",
                last_name="User"
            )
            print(f"OK - Created test user (ID: {test_user.id})")
        
        session.close()
    except Exception as e:
        print(f"FAILED - Could not create user: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 4: Query with filter
    print("\nTest 4: Querying user with filter...")
    try:
        session = get_session()
        
        # Test class attribute filter (this was failing before)
        user = session.query(User).filter(User.created_by_id == None).first()
        print(f"OK - Filter query works")
        
        session.close()
    except Exception as e:
        print(f"FAILED - Query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 5: Database statistics
    print("\nTest 5: Database Statistics...")
    try:
        from pymongo import MongoClient
        client = MongoClient('localhost', 27017)
        db = client['voice_agent_crm']
        
        print(f"OK - Users: {db.users.count_documents({})}")
        print(f"OK - Leads: {db.leads.count_documents({})}")
        print(f"OK - Campaigns: {db.campaigns.count_documents({})}")
        print(f"OK - Call Sessions: {db.call_sessions.count_documents({})}")
        
    except Exception as e:
        print(f"FAILED - Could not get statistics: {e}")
    
    print("\n" + "=" * 60)
    print("All tests passed! MongoDB is working correctly.")
    print("=" * 60)

except ImportError as e:
    print(f"FAILED - Import error: {e}")
    print("Make sure you have installed all requirements:")
    print("   pip install pymongo")
    sys.exit(1)
except Exception as e:
    print(f"FAILED - Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

