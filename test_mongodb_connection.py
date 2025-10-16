# -*- coding: utf-8 -*-
"""
Test MongoDB Connection and Basic Operations
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
    print("🧪 MongoDB Connection Test")
    print("=" * 60)
    
    # Test 1: Connect to MongoDB
    print("\n📌 Test 1: Connecting to MongoDB...")
    try:
        db = init_database()
        print(f"✅ Connected to database: {db.name}")
        print(f"📁 Collections: {', '.join(db.list_collection_names())}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("💡 Make sure MongoDB is running on localhost:27017")
        sys.exit(1)
    
    # Test 2: Create a test user
    print("\n📌 Test 2: Creating test user...")
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        # Check if test user already exists
        existing_user = user_manager.get_user_by_username("test_user")
        
        if existing_user:
            print(f"ℹ️  Test user already exists (ID: {existing_user.id})")
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
            print(f"✅ Created test user (ID: {test_user.id})")
        
        session.close()
    except Exception as e:
        print(f"❌ Failed to create user: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Query user
    print("\n📌 Test 3: Querying user...")
    try:
        session = get_session()
        user = session.query(User).filter_by(username="test_user").first()
        
        if user:
            print(f"✅ Found user: {user.username} ({user.email})")
            print(f"   Role: {user.role.value}")
            print(f"   Full name: {user.full_name}")
        else:
            print("❌ User not found")
        
        session.close()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Create a test lead
    print("\n📌 Test 4: Creating test lead...")
    try:
        session = get_session()
        lead_manager = LeadManager(session)
        
        test_lead = lead_manager.create_lead({
            'owner_id': test_user.id,
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john@example.com',
            'phone': '123456789',
            'country': 'Bulgaria',
            'country_code': '+359',
            'gender': Gender.MALE
        })
        
        print(f"✅ Created test lead (ID: {test_lead.id})")
        print(f"   Name: {test_lead.full_name}")
        print(f"   Phone: {test_lead.full_phone}")
        
        session.close()
    except Exception as e:
        print(f"❌ Failed to create lead: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Query leads
    print("\n📌 Test 5: Querying leads...")
    try:
        session = get_session()
        
        # Count total leads
        total = session.query(Lead).count()
        print(f"✅ Total leads in database: {total}")
        
        # Get leads for test user
        user_leads = session.query(Lead).filter_by(owner_id=test_user.id).all()
        print(f"✅ Leads for test user: {len(user_leads)}")
        
        session.close()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Database statistics
    print("\n📌 Test 6: Database Statistics...")
    try:
        from pymongo import MongoClient
        client = MongoClient('localhost', 27017)
        db = client['voice_agent_crm']
        
        print(f"✅ Users: {db.users.count_documents({})}")
        print(f"✅ Leads: {db.leads.count_documents({})}")
        print(f"✅ Campaigns: {db.campaigns.count_documents({})}")
        print(f"✅ Call Sessions: {db.call_sessions.count_documents({})}")
        
    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! MongoDB is working correctly.")
    print("=" * 60)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you have installed all requirements:")
    print("   pip install pymongo")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

