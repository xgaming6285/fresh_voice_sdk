# -*- coding: utf-8 -*-
"""
Test Campaign Access Fix
"""

import sys

try:
    from crm_database import (
        init_database,
        get_session,
        User,
        UserRole,
        Campaign,
        CampaignManager
    )
    
    print("=" * 60)
    print("Campaign Access Test")
    print("=" * 60)
    
    # Initialize database
    db = init_database()
    
    # Test 1: Create a test campaign
    print("\nTest 1: Creating test campaign...")
    try:
        session = get_session()
        
        # Get an admin user
        admin = session.query(User).filter(User.role == UserRole.ADMIN).first()
        if not admin:
            print("FAILED - No admin user found")
            sys.exit(1)
        
        print(f"OK - Using admin user: {admin.username} (ID: {admin.id})")
        
        # Create a campaign
        campaign_manager = CampaignManager(session)
        test_campaign = campaign_manager.create_campaign(
            owner_id=admin.id,
            name="Test Campaign",
            description="Testing access permissions"
        )
        
        print(f"OK - Created campaign (ID: {test_campaign.id})")
        session.close()
    except Exception as e:
        print(f"FAILED - Could not create campaign: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 2: Query campaigns with filter
    print("\nTest 2: Querying campaigns with owner filter...")
    try:
        session = get_session()
        
        # This is what get_accessible_campaign_ids does
        campaigns = session.query(Campaign).filter(Campaign.owner_id == admin.id).all()
        print(f"OK - Found {len(campaigns)} campaigns for admin")
        
        # Verify our test campaign is in the list
        campaign_ids = [c.id for c in campaigns]
        if test_campaign.id in campaign_ids:
            print(f"OK - Test campaign (ID: {test_campaign.id}) is accessible")
        else:
            print(f"FAILED - Test campaign (ID: {test_campaign.id}) not in accessible list")
            print(f"   Accessible IDs: {campaign_ids}")
            sys.exit(1)
        
        session.close()
    except Exception as e:
        print(f"FAILED - Query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Test get_accessible_campaign_ids function
    print("\nTest 3: Testing get_accessible_campaign_ids function...")
    try:
        from crm_api import get_accessible_campaign_ids
        
        session = get_session()
        accessible_ids = get_accessible_campaign_ids(admin, session)
        
        print(f"OK - Accessible campaign IDs: {accessible_ids}")
        
        if test_campaign.id in accessible_ids:
            print(f"OK - Test campaign (ID: {test_campaign.id}) is in accessible list")
        else:
            print(f"FAILED - Test campaign not accessible through function")
            sys.exit(1)
        
        session.close()
    except Exception as e:
        print(f"FAILED - Function test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All campaign access tests passed!")
    print("=" * 60)
    print("\nThe 403 Forbidden error should now be fixed.")
    print("Restart your application to apply the changes.")

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

