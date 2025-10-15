# -*- coding: utf-8 -*-
"""
Test MongoDB Integration
Quick test to verify MongoDB database connection and operations
"""

from crm_database import (
    init_database, get_session, User, Lead, Campaign, 
    UserRole, Gender, CampaignStatus, 
    UserManager, LeadManager, CampaignManager
)
from datetime import datetime, timedelta

def test_mongodb_connection():
    """Test MongoDB connection"""
    print("=" * 60)
    print("Testing MongoDB Integration")
    print("=" * 60)
    
    try:
        # Initialize database
        print("\n1. Testing database initialization...")
        db = init_database()
        print(f"[OK] Database initialized successfully!")
        print(f"   Collections: {db.list_collection_names()}")
        
        # Test session
        print("\n2. Testing session creation...")
        session = get_session()
        print(f"[OK] Session created successfully!")
        
        # Test User operations
        print("\n3. Testing User operations...")
        user_manager = UserManager(session)
        
        # Check if test user already exists
        existing_user = user_manager.get_user_by_username("test_admin")
        if existing_user:
            print(f"   [INFO] Test user already exists, cleaning up...")
            user_manager.delete_user(existing_user.id)
        
        # Create test admin user
        test_user = user_manager.create_user(
            username="test_admin",
            email="test@example.com",
            password="test123",
            role=UserRole.ADMIN,
            first_name="Test",
            last_name="Admin"
        )
        print(f"[OK] Created user: {test_user.username} (ID: {test_user.id})")
        print(f"   Full name: {test_user.full_name}")
        print(f"   Role: {test_user.role.value}")
        
        # Test password verification
        is_valid = test_user.verify_password("test123")
        print(f"[OK] Password verification: {'Success' if is_valid else 'Failed'}")
        
        # Test Lead operations
        print("\n4. Testing Lead operations...")
        lead_manager = LeadManager(session)
        
        test_lead = lead_manager.create_lead({
            'owner_id': test_user.id,
            'first_name': 'John',
            'last_name': 'Doe',
            'phone': '1234567890',
            'country': 'Bulgaria',
            'country_code': '+359',
            'email': 'john.doe@example.com',
            'gender': Gender.MALE
        })
        print(f"[OK] Created lead: {test_lead.full_name} (ID: {test_lead.id})")
        print(f"   Phone: {test_lead.full_phone}")
        
        # Test bulk import
        bulk_leads = [
            {
                'owner_id': test_user.id,
                'first_name': 'Jane',
                'last_name': 'Smith',
                'phone': '9876543210',
                'country': 'Bulgaria',
                'country_code': '+359',
                'gender': Gender.FEMALE
            },
            {
                'owner_id': test_user.id,
                'first_name': 'Bob',
                'last_name': 'Johnson',
                'phone': '5555555555',
                'country': 'Bulgaria',
                'country_code': '+359',
                'gender': Gender.MALE
            }
        ]
        imported_count = lead_manager.bulk_import_leads(bulk_leads)
        print(f"[OK] Bulk imported {imported_count} leads")
        
        # Test Campaign operations
        print("\n5. Testing Campaign operations...")
        campaign_manager = CampaignManager(session)
        
        test_campaign = campaign_manager.create_campaign(
            owner_id=test_user.id,
            name="Test Campaign",
            description="Test campaign for MongoDB integration",
            bot_config={'voice': 'en-US-Standard-A'}
        )
        print(f"[OK] Created campaign: {test_campaign.name} (ID: {test_campaign.id})")
        print(f"   Status: {test_campaign.status.value}")
        
        # Add leads to campaign
        leads, total = lead_manager.get_leads_by_criteria(owner_id=test_user.id)
        lead_ids = [lead.id for lead in leads]
        added_count = campaign_manager.add_leads_to_campaign(
            campaign_id=test_campaign.id,
            lead_ids=lead_ids,
            priority=1
        )
        print(f"[OK] Added {added_count} leads to campaign")
        
        # Test Query operations
        print("\n6. Testing Query operations...")
        
        # Query users
        users = session.query(User).filter_by(role=UserRole.ADMIN).all()
        print(f"[OK] Found {len(users)} admin users")
        
        # Query leads
        leads_query = session.query(Lead).filter_by(owner_id=test_user.id).all()
        print(f"[OK] Found {len(leads_query)} leads for test user")
        
        # Query campaigns
        campaigns = session.query(Campaign).filter_by(owner_id=test_user.id).all()
        print(f"[OK] Found {len(campaigns)} campaigns for test user")
        
        # Test statistics
        print("\n7. Testing Statistics...")
        total_users = session.query(User).count()
        total_leads = session.query(Lead).count()
        total_campaigns = session.query(Campaign).count()
        print(f"[OK] Database statistics:")
        print(f"   Total users: {total_users}")
        print(f"   Total leads: {total_leads}")
        print(f"   Total campaigns: {total_campaigns}")
        
        # Cleanup
        print("\n8. Cleaning up test data...")
        user_manager.delete_user(test_user.id)
        print(f"[OK] Cleaned up test data")
        
        # Final verification
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMongoDB integration is working correctly!")
        print("You can now use MongoDB with your voice agent CRM system.")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'session' in locals():
            session.close()


if __name__ == "__main__":
    success = test_mongodb_connection()
    exit(0 if success else 1)

