# -*- coding: utf-8 -*-
"""
Clean up duplicate MongoDB collections
This script removes the incorrectly named collections
"""

from pymongo import MongoClient
import os

def cleanup_duplicate_collections():
    """Remove duplicate collections with incorrect names"""
    print("=" * 60)
    print("Cleaning Up Duplicate MongoDB Collections")
    print("=" * 60)
    
    # Connect to MongoDB
    mongo_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGODB_DATABASE', 'voice_agent_crm')
    
    print(f"\nConnecting to: {mongo_url}")
    print(f"Database: {db_name}")
    
    client = MongoClient(mongo_url)
    db = client[db_name]
    
    # List all collections
    print("\nCurrent collections:")
    all_collections = db.list_collection_names()
    for col in sorted(all_collections):
        count = db[col].count_documents({})
        print(f"  - {col}: {count} documents")
    
    # Define collections to remove (incorrect names without underscores)
    collections_to_remove = [
        'callsessions',      # Should be call_sessions
        'campaignleads',     # Should be campaign_leads
        'paymentrequests',   # Should be payment_requests
        'systemsettings',    # Should be system_settings
        'collsessions'       # Typo, should be call_sessions
    ]
    
    # Define correct collection names (keep these)
    correct_collections = [
        'call_sessions',
        'campaign_leads',
        'payment_requests',
        'system_settings',
        'users',
        'leads',
        'campaigns'
    ]
    
    print("\n" + "=" * 60)
    print("Migration Plan:")
    print("=" * 60)
    
    for incorrect_name in collections_to_remove:
        if incorrect_name in all_collections:
            # Find the correct name
            if 'call' in incorrect_name:
                correct_name = 'call_sessions'
            elif 'campaign' in incorrect_name:
                correct_name = 'campaign_leads'
            elif 'payment' in incorrect_name:
                correct_name = 'payment_requests'
            elif 'system' in incorrect_name:
                correct_name = 'system_settings'
            else:
                correct_name = None
            
            incorrect_count = db[incorrect_name].count_documents({})
            
            if correct_name and correct_name in all_collections:
                correct_count = db[correct_name].count_documents({})
                print(f"\n{incorrect_name} ({incorrect_count} docs)")
                print(f"  -> Will merge into: {correct_name} ({correct_count} docs)")
            else:
                print(f"\n{incorrect_name} ({incorrect_count} docs)")
                print(f"  -> Will be dropped")
    
    print("\n" + "=" * 60)
    response = input("\nProceed with cleanup? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("\nCleanup cancelled.")
        return
    
    print("\n" + "=" * 60)
    print("Starting Cleanup...")
    print("=" * 60)
    
    for incorrect_name in collections_to_remove:
        if incorrect_name not in all_collections:
            continue
        
        # Find correct collection name
        if 'call' in incorrect_name:
            correct_name = 'call_sessions'
        elif 'campaign' in incorrect_name:
            correct_name = 'campaign_leads'
        elif 'payment' in incorrect_name:
            correct_name = 'payment_requests'
        elif 'system' in incorrect_name:
            correct_name = 'system_settings'
        else:
            correct_name = None
        
        print(f"\nProcessing: {incorrect_name}")
        
        # If there's a correct collection, try to merge data
        if correct_name and correct_name in all_collections:
            # Get documents from incorrect collection
            docs = list(db[incorrect_name].find())
            
            if docs:
                print(f"  Found {len(docs)} documents to migrate")
                
                # Check for duplicates before inserting
                migrated = 0
                skipped = 0
                
                for doc in docs:
                    # Check if document already exists in correct collection
                    # Use unique identifiers to check
                    if 'id' in doc:
                        existing = db[correct_name].find_one({'id': doc['id']})
                    elif 'session_id' in doc:
                        existing = db[correct_name].find_one({'session_id': doc['session_id']})
                    elif 'username' in doc:
                        existing = db[correct_name].find_one({'username': doc['username']})
                    elif 'key' in doc:
                        existing = db[correct_name].find_one({'key': doc['key']})
                    else:
                        existing = None
                    
                    if not existing:
                        # Insert into correct collection
                        db[correct_name].insert_one(doc)
                        migrated += 1
                    else:
                        skipped += 1
                
                print(f"  Migrated: {migrated} documents")
                print(f"  Skipped (duplicates): {skipped} documents")
        
        # Drop the incorrect collection
        print(f"  Dropping collection: {incorrect_name}")
        db[incorrect_name].drop()
        print(f"  [OK] {incorrect_name} removed")
    
    print("\n" + "=" * 60)
    print("Cleanup Complete!")
    print("=" * 60)
    
    # Show final state
    print("\nFinal collections:")
    final_collections = db.list_collection_names()
    for col in sorted(final_collections):
        count = db[col].count_documents({})
        print(f"  - {col}: {count} documents")
    
    print("\n[SUCCESS] All duplicate collections have been cleaned up!")
    print("Your database now uses consistent naming conventions.")
    
    client.close()


if __name__ == "__main__":
    try:
        cleanup_duplicate_collections()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

