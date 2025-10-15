# -*- coding: utf-8 -*-
"""
Migrate data from SQLite to MongoDB
This script helps migrate existing data from the SQLite database to MongoDB
"""

import sys
from datetime import datetime

def migrate_data():
    """Migrate data from SQLite to MongoDB"""
    print("=" * 60)
    print("SQLite to MongoDB Migration")
    print("=" * 60)
    
    try:
        # Import SQLite database module
        print("\n1. Loading SQLite database...")
        sys.path.insert(0, '.')
        import crm_database_sqlite_backup as sqlite_db
        
        # Import MongoDB database module  
        print("2. Loading MongoDB database...")
        import crm_database as mongo_db
        
        # Initialize both databases
        print("\n3. Initializing databases...")
        sqlite_session = sqlite_db.get_session()
        mongo_session = mongo_db.get_session()
        mongo_database = mongo_db.init_database()
        
        print("✅ Databases initialized")
        
        # Migrate Users
        print("\n4. Migrating Users...")
        sqlite_users = sqlite_session.query(sqlite_db.User).all()
        user_id_mapping = {}  # Old ID -> New ID
        
        for old_user in sqlite_users:
            # Check if user already exists
            existing = mongo_session.query(mongo_db.User).filter_by(username=old_user.username).first()
            if existing:
                print(f"   ⚠️  User {old_user.username} already exists, skipping...")
                user_id_mapping[old_user.id] = existing.id
                continue
            
            new_user = mongo_db.User(
                username=old_user.username,
                email=old_user.email,
                hashed_password=old_user.hashed_password,
                role=old_user.role,
                created_by_id=old_user.created_by_id,
                organization=old_user.organization,
                max_agents=old_user.max_agents,
                subscription_end_date=old_user.subscription_end_date,
                first_name=old_user.first_name,
                last_name=old_user.last_name,
                is_active=old_user.is_active,
                created_at=old_user.created_at,
                updated_at=old_user.updated_at,
                last_login=old_user.last_login
            )
            new_user.save(mongo_database)
            user_id_mapping[old_user.id] = new_user.id
            print(f"   ✅ Migrated user: {old_user.username}")
        
        print(f"✅ Migrated {len(user_id_mapping)} users")
        
        # Update created_by_id references
        print("\n5. Updating user references...")
        for old_id, new_id in user_id_mapping.items():
            user_doc = mongo_database.users.find_one({"id": new_id})
            if user_doc and user_doc.get('created_by_id'):
                old_created_by = user_doc['created_by_id']
                if old_created_by in user_id_mapping:
                    mongo_database.users.update_one(
                        {"id": new_id},
                        {"$set": {"created_by_id": user_id_mapping[old_created_by]}}
                    )
        
        # Migrate Leads
        print("\n6. Migrating Leads...")
        sqlite_leads = sqlite_session.query(sqlite_db.Lead).all()
        lead_id_mapping = {}
        
        for old_lead in sqlite_leads:
            if old_lead.owner_id not in user_id_mapping:
                print(f"   ⚠️  Skipping lead {old_lead.id} - owner not found")
                continue
            
            new_lead = mongo_db.Lead(
                owner_id=user_id_mapping[old_lead.owner_id],
                first_name=old_lead.first_name,
                last_name=old_lead.last_name,
                email=old_lead.email,
                phone=old_lead.phone,
                country=old_lead.country,
                country_code=old_lead.country_code,
                gender=old_lead.gender,
                address=old_lead.address,
                created_at=old_lead.created_at,
                updated_at=old_lead.updated_at,
                last_called_at=old_lead.last_called_at,
                call_count=old_lead.call_count,
                notes=old_lead.notes,
                custom_data=old_lead.custom_data,
                import_batch_id=old_lead.import_batch_id
            )
            new_lead.save(mongo_database)
            lead_id_mapping[old_lead.id] = new_lead.id
        
        print(f"✅ Migrated {len(lead_id_mapping)} leads")
        
        # Migrate Campaigns
        print("\n7. Migrating Campaigns...")
        sqlite_campaigns = sqlite_session.query(sqlite_db.Campaign).all()
        campaign_id_mapping = {}
        
        for old_campaign in sqlite_campaigns:
            if old_campaign.owner_id not in user_id_mapping:
                print(f"   ⚠️  Skipping campaign {old_campaign.id} - owner not found")
                continue
            
            new_campaign = mongo_db.Campaign(
                owner_id=user_id_mapping[old_campaign.owner_id],
                name=old_campaign.name,
                description=old_campaign.description,
                status=old_campaign.status,
                bot_config=old_campaign.bot_config,
                dialing_config=old_campaign.dialing_config,
                schedule_config=old_campaign.schedule_config,
                total_leads=old_campaign.total_leads,
                leads_called=old_campaign.leads_called,
                leads_answered=old_campaign.leads_answered,
                leads_rejected=old_campaign.leads_rejected,
                leads_failed=old_campaign.leads_failed,
                created_at=old_campaign.created_at,
                updated_at=old_campaign.updated_at,
                started_at=old_campaign.started_at,
                completed_at=old_campaign.completed_at
            )
            new_campaign.save(mongo_database)
            campaign_id_mapping[old_campaign.id] = new_campaign.id
        
        print(f"✅ Migrated {len(campaign_id_mapping)} campaigns")
        
        # Migrate Campaign Leads
        print("\n8. Migrating Campaign Leads...")
        sqlite_campaign_leads = sqlite_session.query(sqlite_db.CampaignLead).all()
        campaign_lead_count = 0
        
        for old_cl in sqlite_campaign_leads:
            if old_cl.campaign_id not in campaign_id_mapping:
                continue
            if old_cl.lead_id not in lead_id_mapping:
                continue
            
            new_cl = mongo_db.CampaignLead(
                campaign_id=campaign_id_mapping[old_cl.campaign_id],
                lead_id=lead_id_mapping[old_cl.lead_id],
                status=old_cl.status,
                priority=old_cl.priority,
                call_attempts=old_cl.call_attempts,
                last_attempt_at=old_cl.last_attempt_at,
                scheduled_for=old_cl.scheduled_for,
                call_session_id=old_cl.call_session_id,
                call_duration=old_cl.call_duration,
                call_result=old_cl.call_result,
                call_notes=old_cl.call_notes,
                added_at=old_cl.added_at,
                called_at=old_cl.called_at,
                completed_at=old_cl.completed_at
            )
            new_cl.save(mongo_database)
            campaign_lead_count += 1
        
        print(f"✅ Migrated {campaign_lead_count} campaign leads")
        
        # Migrate Call Sessions
        print("\n9. Migrating Call Sessions...")
        sqlite_sessions = sqlite_session.query(sqlite_db.CallSession).all()
        session_count = 0
        
        for old_session in sqlite_sessions:
            if old_session.owner_id and old_session.owner_id not in user_id_mapping:
                continue
            
            new_session = mongo_db.CallSession(
                session_id=old_session.session_id,
                campaign_id=campaign_id_mapping.get(old_session.campaign_id) if old_session.campaign_id else None,
                lead_id=lead_id_mapping.get(old_session.lead_id) if old_session.lead_id else None,
                owner_id=user_id_mapping.get(old_session.owner_id) if old_session.owner_id else None,
                caller_id=old_session.caller_id,
                called_number=old_session.called_number,
                status=old_session.status,
                started_at=old_session.started_at,
                answered_at=old_session.answered_at,
                ended_at=old_session.ended_at,
                duration=old_session.duration,
                talk_time=old_session.talk_time,
                recording_path=old_session.recording_path,
                transcript_status=old_session.transcript_status,
                transcript_language=old_session.transcript_language,
                sentiment_score=old_session.sentiment_score,
                interest_level=old_session.interest_level,
                key_points=old_session.key_points,
                follow_up_required=old_session.follow_up_required,
                follow_up_notes=old_session.follow_up_notes,
                call_metadata=old_session.call_metadata,
                created_at=old_session.created_at,
                updated_at=old_session.updated_at
            )
            new_session.save(mongo_database)
            session_count += 1
        
        print(f"✅ Migrated {session_count} call sessions")
        
        # Migrate Payment Requests
        print("\n10. Migrating Payment Requests...")
        sqlite_payments = sqlite_session.query(sqlite_db.PaymentRequest).all()
        payment_count = 0
        
        for old_payment in sqlite_payments:
            if old_payment.admin_id not in user_id_mapping:
                continue
            
            new_payment = mongo_db.PaymentRequest(
                admin_id=user_id_mapping[old_payment.admin_id],
                num_agents=old_payment.num_agents,
                total_amount=old_payment.total_amount,
                status=old_payment.status,
                payment_notes=old_payment.payment_notes,
                admin_notes=old_payment.admin_notes,
                created_at=old_payment.created_at,
                updated_at=old_payment.updated_at,
                approved_at=old_payment.approved_at,
                approved_by_id=user_id_mapping.get(old_payment.approved_by_id) if old_payment.approved_by_id else None
            )
            new_payment.save(mongo_database)
            payment_count += 1
        
        print(f"✅ Migrated {payment_count} payment requests")
        
        # Migrate System Settings
        print("\n11. Migrating System Settings...")
        sqlite_settings = sqlite_session.query(sqlite_db.SystemSettings).all()
        settings_count = 0
        
        for old_setting in sqlite_settings:
            new_setting = mongo_db.SystemSettings(
                key=old_setting.key,
                value=old_setting.value,
                updated_at=old_setting.updated_at,
                updated_by_id=user_id_mapping.get(old_setting.updated_by_id) if old_setting.updated_by_id else None
            )
            new_setting.save(mongo_database)
            settings_count += 1
        
        print(f"✅ Migrated {settings_count} system settings")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nMigration Summary:")
        print(f"  Users:           {len(user_id_mapping)}")
        print(f"  Leads:           {len(lead_id_mapping)}")
        print(f"  Campaigns:       {len(campaign_id_mapping)}")
        print(f"  Campaign Leads:  {campaign_lead_count}")
        print(f"  Call Sessions:   {session_count}")
        print(f"  Payment Requests:{payment_count}")
        print(f"  System Settings: {settings_count}")
        print("\nYour data has been successfully migrated to MongoDB!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'sqlite_session' in locals():
            sqlite_session.close()
        if 'mongo_session' in locals():
            mongo_session.close()


if __name__ == "__main__":
    print("\n⚠️  WARNING: This will migrate all data from SQLite to MongoDB.")
    print("Make sure MongoDB is running on localhost:27017")
    print("The SQLite database backup is available at: voice_agent_crm_sqlite_backup.db")
    
    response = input("\nDo you want to proceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        success = migrate_data()
        exit(0 if success else 1)
    else:
        print("\nMigration cancelled.")
        exit(0)

