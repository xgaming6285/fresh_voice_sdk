# -*- coding: utf-8 -*-
"""
Migration Script: SQLite to MongoDB
Transfers all data from voice_agent_crm.db (SQLite) to MongoDB
"""

import os
import sys
from datetime import datetime
from pymongo import MongoClient

# Import SQLite models
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Temporarily import the SQLite version
sys.path.insert(0, os.path.dirname(__file__))
from crm_database_sqlite_backup import (
    Base as SQLiteBase,
    User as SQLiteUser,
    Lead as SQLiteLead,
    Campaign as SQLiteCampaign,
    CampaignLead as SQLiteCampaignLead,
    CallSession as SQLiteCallSession,
    PaymentRequest as SQLitePaymentRequest,
    SystemSettings as SQLiteSystemSettings,
    UserRole, Gender, CampaignStatus, CallStatus, PaymentRequestStatus
)

# Import MongoDB implementation
from crm_database_mongodb import MongoDB, init_database as init_mongodb

def get_sqlite_session():
    """Get SQLite session"""
    sqlite_db_path = os.getenv('SQLITE_DATABASE', 'voice_agent_crm.db')
    
    if not os.path.exists(sqlite_db_path):
        print(f"❌ SQLite database not found: {sqlite_db_path}")
        return None
    
    engine = create_engine(f'sqlite:///{sqlite_db_path}')
    Session = sessionmaker(bind=engine)
    return Session()

def convert_enum_to_value(obj):
    """Convert enum to string value"""
    if isinstance(obj, (UserRole, Gender, CampaignStatus, CallStatus, PaymentRequestStatus)):
        return obj.value
    return obj

def migrate_users(sqlite_session, mongodb):
    """Migrate users from SQLite to MongoDB"""
    print("\n📊 Migrating users...")
    
    try:
        users = sqlite_session.query(SQLiteUser).all()
        
        if not users:
            print("  ℹ️  No users to migrate")
            return 0
        
        mongo_users = []
        for user in users:
            user_doc = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "hashed_password": user.hashed_password,
                "role": convert_enum_to_value(user.role),
                "created_by_id": user.created_by_id,
                "organization": user.organization,
                "max_agents": user.max_agents,
                "subscription_end_date": user.subscription_end_date,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "last_login": user.last_login
            }
            mongo_users.append(user_doc)
        
        # Clear existing users and insert
        mongodb.users.delete_many({})
        mongodb.users.insert_many(mongo_users)
        
        print(f"  ✅ Migrated {len(mongo_users)} users")
        return len(mongo_users)
    
    except Exception as e:
        print(f"  ❌ Error migrating users: {e}")
        import traceback
        traceback.print_exc()
        return 0

def migrate_leads(sqlite_session, mongodb):
    """Migrate leads from SQLite to MongoDB"""
    print("\n📊 Migrating leads...")
    
    try:
        leads = sqlite_session.query(SQLiteLead).all()
        
        if not leads:
            print("  ℹ️  No leads to migrate")
            return 0
        
        mongo_leads = []
        for lead in leads:
            lead_doc = {
                "id": lead.id,
                "owner_id": lead.owner_id,
                "first_name": lead.first_name,
                "last_name": lead.last_name,
                "email": lead.email,
                "phone": lead.phone,
                "country": lead.country,
                "country_code": lead.country_code,
                "gender": convert_enum_to_value(lead.gender),
                "address": lead.address,
                "created_at": lead.created_at,
                "updated_at": lead.updated_at,
                "last_called_at": lead.last_called_at,
                "call_count": lead.call_count,
                "notes": lead.notes,
                "custom_data": lead.custom_data,
                "import_batch_id": lead.import_batch_id
            }
            mongo_leads.append(lead_doc)
        
        # Clear existing leads and insert
        mongodb.leads.delete_many({})
        mongodb.leads.insert_many(mongo_leads)
        
        print(f"  ✅ Migrated {len(mongo_leads)} leads")
        return len(mongo_leads)
    
    except Exception as e:
        print(f"  ❌ Error migrating leads: {e}")
        import traceback
        traceback.print_exc()
        return 0

def migrate_campaigns(sqlite_session, mongodb):
    """Migrate campaigns from SQLite to MongoDB"""
    print("\n📊 Migrating campaigns...")
    
    try:
        campaigns = sqlite_session.query(SQLiteCampaign).all()
        
        if not campaigns:
            print("  ℹ️  No campaigns to migrate")
            return 0
        
        mongo_campaigns = []
        for campaign in campaigns:
            campaign_doc = {
                "id": campaign.id,
                "owner_id": campaign.owner_id,
                "name": campaign.name,
                "description": campaign.description,
                "status": convert_enum_to_value(campaign.status),
                "bot_config": campaign.bot_config,
                "dialing_config": campaign.dialing_config,
                "schedule_config": campaign.schedule_config,
                "total_leads": campaign.total_leads,
                "leads_called": campaign.leads_called,
                "leads_answered": campaign.leads_answered,
                "leads_rejected": campaign.leads_rejected,
                "leads_failed": campaign.leads_failed,
                "created_at": campaign.created_at,
                "updated_at": campaign.updated_at,
                "started_at": campaign.started_at,
                "completed_at": campaign.completed_at
            }
            mongo_campaigns.append(campaign_doc)
        
        # Clear existing campaigns and insert
        mongodb.campaigns.delete_many({})
        mongodb.campaigns.insert_many(mongo_campaigns)
        
        print(f"  ✅ Migrated {len(mongo_campaigns)} campaigns")
        return len(mongo_campaigns)
    
    except Exception as e:
        print(f"  ❌ Error migrating campaigns: {e}")
        import traceback
        traceback.print_exc()
        return 0

def migrate_campaign_leads(sqlite_session, mongodb):
    """Migrate campaign_leads from SQLite to MongoDB"""
    print("\n📊 Migrating campaign leads...")
    
    try:
        campaign_leads = sqlite_session.query(SQLiteCampaignLead).all()
        
        if not campaign_leads:
            print("  ℹ️  No campaign leads to migrate")
            return 0
        
        mongo_campaign_leads = []
        for cl in campaign_leads:
            cl_doc = {
                "id": cl.id,
                "campaign_id": cl.campaign_id,
                "lead_id": cl.lead_id,
                "status": convert_enum_to_value(cl.status),
                "priority": cl.priority,
                "call_attempts": cl.call_attempts,
                "last_attempt_at": cl.last_attempt_at,
                "scheduled_for": cl.scheduled_for,
                "call_session_id": cl.call_session_id,
                "call_duration": cl.call_duration,
                "call_result": cl.call_result,
                "call_notes": cl.call_notes,
                "added_at": cl.added_at,
                "called_at": cl.called_at,
                "completed_at": cl.completed_at
            }
            mongo_campaign_leads.append(cl_doc)
        
        # Clear existing campaign_leads and insert
        mongodb.campaign_leads.delete_many({})
        mongodb.campaign_leads.insert_many(mongo_campaign_leads)
        
        print(f"  ✅ Migrated {len(mongo_campaign_leads)} campaign leads")
        return len(mongo_campaign_leads)
    
    except Exception as e:
        print(f"  ❌ Error migrating campaign leads: {e}")
        import traceback
        traceback.print_exc()
        return 0

def migrate_call_sessions(sqlite_session, mongodb):
    """Migrate call sessions from SQLite to MongoDB"""
    print("\n📊 Migrating call sessions...")
    
    try:
        sessions = sqlite_session.query(SQLiteCallSession).all()
        
        if not sessions:
            print("  ℹ️  No call sessions to migrate")
            return 0
        
        mongo_sessions = []
        for session in sessions:
            session_doc = {
                "id": session.id,
                "session_id": session.session_id,
                "campaign_id": session.campaign_id,
                "lead_id": session.lead_id,
                "owner_id": session.owner_id,
                "caller_id": session.caller_id,
                "called_number": session.called_number,
                "status": convert_enum_to_value(session.status),
                "started_at": session.started_at,
                "answered_at": session.answered_at,
                "ended_at": session.ended_at,
                "duration": session.duration,
                "talk_time": session.talk_time,
                "recording_path": session.recording_path,
                "transcript_status": session.transcript_status,
                "transcript_language": session.transcript_language,
                "sentiment_score": session.sentiment_score,
                "interest_level": session.interest_level,
                "key_points": session.key_points,
                "follow_up_required": session.follow_up_required,
                "follow_up_notes": session.follow_up_notes,
                "call_metadata": session.call_metadata,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            mongo_sessions.append(session_doc)
        
        # Clear existing call_sessions and insert
        mongodb.call_sessions.delete_many({})
        mongodb.call_sessions.insert_many(mongo_sessions)
        
        print(f"  ✅ Migrated {len(mongo_sessions)} call sessions")
        return len(mongo_sessions)
    
    except Exception as e:
        print(f"  ❌ Error migrating call sessions: {e}")
        import traceback
        traceback.print_exc()
        return 0

def migrate_payment_requests(sqlite_session, mongodb):
    """Migrate payment requests from SQLite to MongoDB"""
    print("\n📊 Migrating payment requests...")
    
    try:
        requests = sqlite_session.query(SQLitePaymentRequest).all()
        
        if not requests:
            print("  ℹ️  No payment requests to migrate")
            return 0
        
        mongo_requests = []
        for req in requests:
            req_doc = {
                "id": req.id,
                "admin_id": req.admin_id,
                "num_agents": req.num_agents,
                "total_amount": req.total_amount,
                "status": convert_enum_to_value(req.status),
                "payment_notes": req.payment_notes,
                "admin_notes": req.admin_notes,
                "created_at": req.created_at,
                "updated_at": req.updated_at,
                "approved_at": req.approved_at,
                "approved_by_id": req.approved_by_id
            }
            mongo_requests.append(req_doc)
        
        # Clear existing payment_requests and insert
        mongodb.payment_requests.delete_many({})
        mongodb.payment_requests.insert_many(mongo_requests)
        
        print(f"  ✅ Migrated {len(mongo_requests)} payment requests")
        return len(mongo_requests)
    
    except Exception as e:
        print(f"  ❌ Error migrating payment requests: {e}")
        import traceback
        traceback.print_exc()
        return 0

def migrate_system_settings(sqlite_session, mongodb):
    """Migrate system settings from SQLite to MongoDB"""
    print("\n📊 Migrating system settings...")
    
    try:
        settings = sqlite_session.query(SQLiteSystemSettings).all()
        
        if not settings:
            print("  ℹ️  No system settings to migrate")
            return 0
        
        mongo_settings = []
        for setting in settings:
            setting_doc = {
                "id": setting.id,
                "key": setting.key,
                "value": setting.value,
                "updated_at": setting.updated_at,
                "updated_by_id": setting.updated_by_id
            }
            mongo_settings.append(setting_doc)
        
        # Clear existing system_settings and insert
        mongodb.system_settings.delete_many({})
        mongodb.system_settings.insert_many(mongo_settings)
        
        print(f"  ✅ Migrated {len(mongo_settings)} system settings")
        return len(mongo_settings)
    
    except Exception as e:
        print(f"  ❌ Error migrating system settings: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Main migration function"""
    print("=" * 60)
    print("🔄 SQLite to MongoDB Migration Script")
    print("=" * 60)
    
    # Connect to SQLite
    print("\n📂 Connecting to SQLite database...")
    sqlite_session = get_sqlite_session()
    
    if not sqlite_session:
        print("❌ Failed to connect to SQLite database")
        return
    
    print("✅ Connected to SQLite")
    
    # Connect to MongoDB
    print("\n📂 Connecting to MongoDB...")
    try:
        mongodb = init_mongodb()
        print(f"✅ Connected to MongoDB (database: {mongodb.name})")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("💡 Make sure MongoDB is running on localhost:27017")
        return
    
    # Perform migration
    print("\n🚀 Starting migration...")
    print("-" * 60)
    
    total_migrated = 0
    
    # Migrate in order (respecting foreign key relationships)
    total_migrated += migrate_users(sqlite_session, mongodb)
    total_migrated += migrate_leads(sqlite_session, mongodb)
    total_migrated += migrate_campaigns(sqlite_session, mongodb)
    total_migrated += migrate_campaign_leads(sqlite_session, mongodb)
    total_migrated += migrate_call_sessions(sqlite_session, mongodb)
    total_migrated += migrate_payment_requests(sqlite_session, mongodb)
    total_migrated += migrate_system_settings(sqlite_session, mongodb)
    
    print("\n" + "=" * 60)
    print(f"✅ Migration completed!")
    print(f"📊 Total records migrated: {total_migrated}")
    print("=" * 60)
    
    # Verify migration
    print("\n🔍 Verifying migration...")
    print(f"  Users: {mongodb.users.count_documents({})}")
    print(f"  Leads: {mongodb.leads.count_documents({})}")
    print(f"  Campaigns: {mongodb.campaigns.count_documents({})}")
    print(f"  Campaign Leads: {mongodb.campaign_leads.count_documents({})}")
    print(f"  Call Sessions: {mongodb.call_sessions.count_documents({})}")
    print(f"  Payment Requests: {mongodb.payment_requests.count_documents({})}")
    print(f"  System Settings: {mongodb.system_settings.count_documents({})}")
    
    # Close SQLite session
    sqlite_session.close()
    
    print("\n✅ Migration complete! You can now use MongoDB for your CRM.")
    print("💡 The original SQLite database has been preserved.")

if __name__ == "__main__":
    main()

