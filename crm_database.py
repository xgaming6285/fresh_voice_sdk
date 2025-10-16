# -*- coding: utf-8 -*-
"""
CRM Database Models for Voice Agent
Now using MongoDB instead of SQLite/SQLAlchemy

This file maintains backward compatibility by importing from the MongoDB implementation.
The original SQLite implementation has been backed up to crm_database_sqlite_backup.py
"""

# Import everything from MongoDB implementation
from crm_database_mongodb import (
# Enums
    UserRole,
    Gender,
    CampaignStatus,
    CallStatus,
    PaymentRequestStatus,
    
    # Models
    User,
    Lead,
    Campaign,
    CampaignLead,
    CallSession,
    PaymentRequest,
    SystemSettings,
    
    # Managers
    UserManager,
    LeadManager,
    CampaignManager,
    
    # Database functions
    init_database,
    get_session,
    
    # MongoDB connection
    MongoDB,
    
    # Password context for compatibility
    pwd_context
)

# For backward compatibility with any code that imports Base
Base = None  # MongoDB doesn't use SQLAlchemy Base

# Mock SQLAlchemy functions for compatibility
def joinedload(*args, **kwargs):
    """Mock joinedload for SQLAlchemy compatibility - MongoDB doesn't need it"""
    return None

if __name__ == "__main__":
    # Initialize MongoDB database
    db = init_database()
    print("✅ MongoDB database initialized successfully!")
    print(f"📊 Database: {db.name}")
    print(f"📁 Collections: {db.list_collection_names()}")
