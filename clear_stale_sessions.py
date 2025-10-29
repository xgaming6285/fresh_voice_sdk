"""
Utility script to manually clear stale active sessions
Run this if the UI shows active sessions that ended long ago
"""
import requests

def clear_stale_sessions():
    """Check and report on stale sessions that can be manually cleared"""
    try:
        # Get active sessions from the API
        response = requests.get("http://localhost:8000/api/sessions")
        if response.status_code == 200:
            data = response.json()
            sessions = data.get("sessions", [])
            
            if not sessions:
                print("✅ No active sessions found")
                return
            
            print(f"\n⚠️  Found {len(sessions)} active session(s):\n")
            
            for session in sessions:
                session_id = session.get("session_id")
                duration = session.get("duration_seconds", 0)
                caller_id = session.get("caller_id")
                called_number = session.get("called_number")
                
                print(f"Session ID: {session_id}")
                print(f"  Caller: {caller_id} → {called_number}")
                print(f"  Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
                print(f"  Status: {session.get('status')}")
                
                # If session is longer than 10 minutes, it's likely stale
                if duration > 600:
                    print(f"  ⚠️  This session appears STALE (running for {duration/60:.1f} minutes)")
                
                print()
            
            print("\n" + "="*60)
            print("MANUAL FIX:")
            print("="*60)
            print("\n1. The stale session cleanup fix has been applied to the code.")
            print("2. Restart the voice agent for the fix to take effect:")
            print("   - Stop the current process (Ctrl+C)")
            print("   - Run: python windows_voice_agent.py")
            print("\n3. The new code includes:")
            print("   - Enhanced logging to debug session removal")
            print("   - Better asyncio task cleanup to prevent errors")
            print("   - Safety checks to ensure sessions are removed")
            print("\nNOTE: Existing stale sessions will be cleared on next call or restart.")
            
        else:
            print(f"❌ Error: Failed to get sessions (status code: {response.status_code})")
    
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to the voice agent API")
        print("Make sure the voice agent is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    clear_stale_sessions()

