# -*- coding: utf-8 -*-
"""
Delete Conversion Records
Removes all sessions marked as 'interested' from MongoDB
"""

from pymongo import MongoClient

def delete_conversions():
    """Delete all conversion records (interested sessions)"""
    try:
        print("=" * 70)
        print("Delete Conversion Records")
        print("=" * 70)
        
        # Connect to MongoDB
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        db = client['voice_agent_crm']
        
        # Test connection
        client.admin.command('ping')
        print(f"\nConnected to MongoDB: {db.name}")
        
        # Count before deletion
        before_count = db.call_sessions.count_documents({'analysis.status': 'interested'})
        
        print(f"\nFound {before_count} conversion records (interested sessions)")
        
        if before_count == 0:
            print("No conversion records to delete.")
            return
        
        # Delete conversion records
        print(f"\nDeleting {before_count} conversion records...")
        result = db.call_sessions.delete_many({'analysis.status': 'interested'})
        
        print(f"[OK] Deleted {result.deleted_count} conversion records")
        
        # Show remaining counts
        print("\nRemaining sessions:")
        print(f"  - Total sessions: {db.call_sessions.count_documents({})}")
        print(f"  - Interested (conversions): {db.call_sessions.count_documents({'analysis.status': 'interested'})}")
        print(f"  - Not interested: {db.call_sessions.count_documents({'analysis.status': 'not_interested'})}")
        
        print("\n" + "=" * 70)
        print("Conversion records deleted successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    delete_conversions()

