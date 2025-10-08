"""
Migration script to add owner_id to call_sessions table
Run this after updating the CallSession model
"""
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    """Add owner_id column to call_sessions table"""
    try:
        conn = sqlite3.connect('voice_agent_crm.db')
        cursor = conn.cursor()
        
        # Check if owner_id column already exists
        cursor.execute("PRAGMA table_info(call_sessions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'owner_id' in columns:
            logger.info("✅ owner_id column already exists in call_sessions table")
            conn.close()
            return
        
        logger.info("Adding owner_id column to call_sessions table...")
        
        # Add owner_id column (nullable for now to allow existing records)
        cursor.execute("""
            ALTER TABLE call_sessions 
            ADD COLUMN owner_id INTEGER
        """)
        
        # Get the first admin user to assign to existing sessions
        cursor.execute("""
            SELECT id FROM users 
            WHERE role = 'admin' 
            ORDER BY id ASC 
            LIMIT 1
        """)
        admin_result = cursor.fetchone()
        
        if admin_result:
            admin_id = admin_result[0]
            # Update existing sessions to belong to first admin
            cursor.execute("""
                UPDATE call_sessions 
                SET owner_id = ? 
                WHERE owner_id IS NULL
            """, (admin_id,))
            logger.info(f"✅ Assigned {cursor.rowcount} existing call sessions to admin ID {admin_id}")
        else:
            logger.warning("⚠️ No admin users found. Existing call sessions have no owner.")
        
        conn.commit()
        logger.info("✅ Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    migrate()

