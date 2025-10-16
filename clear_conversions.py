# -*- coding: utf-8 -*-
"""
Clear Conversion Records (Interested Sessions)
Removes call sessions with 'interested' status from MongoDB
"""

from pymongo import MongoClient

def clear_conversions():
    """Clear all sessions marked as 'interested' (conversions)"""
    try:
        print("=" * 70)
        print("Clear Conversion Records")
        print("=" * 70)
        
        # Connect to MongoDB
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        db = client['voice_agent_crm']
        
        # Test connection
        client.admin.command('ping')
        print(f"\nConnected to MongoDB: {db.name}")
        
        # Count sessions with 'interested' status
        interested_count = db.call_sessions.count_documents({
            'analysis.status': 'interested'
        })
        
        print(f"\nFound {interested_count} conversion records (interested sessions)")
        
        if interested_count == 0:
            print("No conversion records to delete.")
            return
        
        # Ask for confirmation
        print("\nOptions:")
        print("1. Delete ONLY conversion records (sessions with 'interested' status)")
        print("2. Delete ALL call sessions (complete reset)")
        print("3. Cancel (do nothing)")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            # Delete only interested sessions
            result = db.call_sessions.delete_many({
                'analysis.status': 'interested'
            })
            print(f"\n✅ Deleted {result.deleted_count} conversion records (interested sessions)")
            
        elif choice == '2':
            # Delete all sessions
            total_count = db.call_sessions.count_documents({})
            print(f"\nThis will delete ALL {total_count} call sessions!")
            confirm = input("Are you sure? Type 'YES' to confirm: ").strip()
            
            if confirm == 'YES':
                result = db.call_sessions.delete_many({})
                print(f"\n✅ Deleted {result.deleted_count} call sessions")
            else:
                print("\n❌ Cancelled")
                
        else:
            print("\n❌ Cancelled")
            return
        
        # Show remaining counts
        print("\nRemaining records:")
        print(f"  - Total sessions: {db.call_sessions.count_documents({})}")
        print(f"  - Interested (conversions): {db.call_sessions.count_documents({'analysis.status': 'interested'})}")
        print(f"  - Not interested: {db.call_sessions.count_documents({'analysis.status': 'not_interested'})}")
        
        print("\n" + "=" * 70)
        print("Operation complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_conversions()

