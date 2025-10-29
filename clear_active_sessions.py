"""
Emergency script to clear stale active sessions
Run this after restarting the voice agent if old sessions are still showing as active
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from windows_voice_agent import active_sessions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_stale_sessions():
    """Clear all stale sessions from active_sessions dictionary"""
    if not active_sessions:
        logger.info("✅ No active sessions to clear")
        return
    
    logger.info(f"⚠️  Found {len(active_sessions)} session(s) in active_sessions:")
    
    for session_id in list(active_sessions.keys()):
        logger.info(f"  - Removing session: {session_id}")
        del active_sessions[session_id]
    
    logger.info(f"✅ Cleared {len(list(active_sessions.keys()))} session(s)")
    logger.info(f"📊 Active sessions remaining: {len(active_sessions)}")

if __name__ == "__main__":
    clear_stale_sessions()

