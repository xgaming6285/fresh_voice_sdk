# -*- coding: utf-8 -*-
"""
Auto-clean duplicate MongoDB collections (non-interactive)
This script automatically removes incorrectly named collections and merges data
"""

from pymongo import MongoClient
import os

def auto_cleanup_duplicate_collections():
    """Remove duplicate collections with incorrect names (automatic)"""
    print("=" * 60)
    print("Auto-Cleaning Duplicate MongoDB Collections")
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
        'collsessions',      # Typo, should be call_sessions
        'campaignleads',     # Should be campaign_leads
        'paymentrequests',   # Should be payment_requests
        'systemsettingss',   # Should be system_settings
    ]
    
    print("\n" + "=" * 60)
    print("Starting Cleanup...")
    print("=" * 60)
    
    total_migrated = 0
    total_skipped = 0
    
    for incorrect_name in collections_to_remove:
        if incorrect_name not in all_collections:
            continue
        
        # Find correct collection name
        if 'call' in incorrect_name or 'coll' in incorrect_name:
            correct_name = 'call_sessions'
        elif 'campaign' in incorrect_name:
            correct_name = 'campaign_leads'
        elif 'payment' in incorrect_name:
            correct_name = 'payment_requests'
        elif 'system' in incorrect_name:
            correct_name = 'system_settings'
        else:
            correct_name = None
        
        print(f"\n[Processing] {incorrect_name}")
        
        # If there's a correct collection, try to merge data
        if correct_name:
            # Ensure correct collection exists
            if correct_name not in all_collections:
                db.create_collection(correct_name)
                print(f"  Created: {correct_name}")
            
            # Get documents from incorrect collection
            docs = list(db[incorrect_name].find())
            
            if docs:
                print(f"  Found {len(docs)} documents to migrate")
                
                # Migrate documents
                migrated = 0
                skipped = 0
                
                for doc in docs:
                    # Check if document already exists in correct collection
                    # Use unique identifiers to check
                    existing = None
                    
                    if 'id' in doc:
                        existing = db[correct_name].find_one({'id': doc['id']})
                    elif 'session_id' in doc:
                        existing = db[correct_name].find_one({'session_id': doc['session_id']})
                    elif 'username' in doc:
                        existing = db[correct_name].find_one({'username': doc['username']})
                    elif 'key' in doc:
                        existing = db[correct_name].find_one({'key': doc['key']})
                    elif '_id' in doc:
                        existing = db[correct_name].find_one({'_id': doc['_id']})
                    
                    if not existing:
                        # Insert into correct collection
                        try:
                            db[correct_name].insert_one(doc)
                            migrated += 1
                        except Exception as e:
                            print(f"    Warning: Could not migrate document: {e}")
                            skipped += 1
                    else:
                        skipped += 1
                
                print(f"  Migrated: {migrated} documents")
                if skipped > 0:
                    print(f"  Skipped (duplicates): {skipped} documents")
                
                total_migrated += migrated
                total_skipped += skipped
        
        # Drop the incorrect collection
        print(f"  Dropping collection: {incorrect_name}")
        db[incorrect_name].drop()
        print(f"  [OK] {incorrect_name} removed")
    
    print("\n" + "=" * 60)
    print("Cleanup Complete!")
    print("=" * 60)
    print(f"\nTotal documents migrated: {total_migrated}")
    if total_skipped > 0:
        print(f"Total duplicates skipped: {total_skipped}")
    
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
        auto_cleanup_duplicate_collections()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

