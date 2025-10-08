"""
Script to clean up call sessions that have no owner (owner_id = NULL)
These are orphaned sessions from before the ownership system was implemented
"""
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup():
    """Delete call sessions with no owner"""
    try:
        conn = sqlite3.connect('voice_agent_crm.db')
        cursor = conn.cursor()
        
        # Check how many orphaned sessions exist
        cursor.execute("""
            SELECT COUNT(*) FROM call_sessions 
            WHERE owner_id IS NULL
        """)
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("✅ No orphaned call sessions found. Database is clean!")
            conn.close()
            return
        
        logger.warning(f"⚠️ Found {count} orphaned call sessions (no owner)")
        
        # Delete orphaned sessions
        cursor.execute("""
            DELETE FROM call_sessions 
            WHERE owner_id IS NULL
        """)
        
        deleted = cursor.rowcount
        conn.commit()
        
        logger.info(f"✅ Deleted {deleted} orphaned call sessions")
        logger.info("✅ Cleanup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    cleanup()

