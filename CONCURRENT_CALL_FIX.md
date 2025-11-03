# Concurrent Call Isolation - FINAL FIX

## Problem

When running concurrent calls (agent4 + admin2), the system couldn't handle both calls simultaneously:
- RTP packets constantly switched between sessions
- Audio lagged and mixed between users
- One call hung up after ~10 seconds
- The other stayed active but disconnected from Gemini

## Root Cause

**Both sessions initialized with the same default port:**
```python
session1.remote_addr = ('192.168.50.50', 5004)  # Default port
session2.remote_addr = ('192.168.50.50', 5004)  # Same default!
```

When RTP packets arrived:
1. First packet from call 1 (port 18920) arrives
2. Loop checks: "Which session has port 5004?" → **BOTH match!**
3. **FIRST session in dict gets initialized** (might be wrong one!)
4. Second packet from call 2 (port 19508) arrives
5. No match found, tries to initialize again
6. **Constant switching between ports**

## Solution

### 1. RTP Session Isolation with Initialization Flags

**New matching strategy:**
```python
uninitialized_session = None

for session_id, rtp_session in sessions.items():
    # Priority 1: Match by SSRC (if already locked)
    if rtp_session.remote_ssrc == ssrc:
        return rtp_session  # ✅ Perfect match
    
    # Priority 2: Match by exact (IP, port)
    if rtp_session.remote_addr == addr:
        rtp_session.remote_ssrc = ssrc
        return rtp_session  # ✅ Good match
    
    # Priority 3: Track FIRST uninitialized session
    if not hasattr(rtp_session, 'rtp_initialized') and rtp_session.remote_addr[1] == 5004:
        if not hasattr(rtp_session, 'remote_ssrc'):
            if uninitialized_session is None:
                uninitialized_session = rtp_session  # Only take first!

# Initialize ONLY the first uninitialized session
if uninitialized_session:
    uninitialized_session.remote_addr = addr
    uninitialized_session.remote_ssrc = ssrc
    uninitialized_session.rtp_initialized = True  # Mark as claimed
```

**Key improvement:**
- ✅ Only ONE session can claim a new port/SSRC combination
- ✅ `rtp_initialized` flag prevents re-initialization
- ✅ SSRC locking ensures packets stay with correct session
- ✅ First-come-first-serve for uninitialized sessions

### 2. Google API Key Isolation

Each user gets their own API key:

```python
# User assigned API key in database
user.google_api_key = "AIzaSy..."

# Each session creates its own Gemini client
self.session_voice_client = genai.Client(
    api_key=user.google_api_key  # User-specific key!
)

# Completely isolated:
# - Separate WebSocket to Gemini
# - Independent audio streams
# - No state sharing
```

## Results

### Before Fix:
```
📍 Updating RTP address from ('192.168.50.50', 19508) to ('192.168.50.50', 18920)
📍 Updating RTP address from ('192.168.50.50', 18920) to ('192.168.50.50', 19508)
...
[Call 1 hangs up after 10s]
[Call 2 stays active but audio doesn't work]
```

### After Fix:
```
📍 Initializing RTP session SESSION_1 with actual port 10864 and SSRC 478356044
🔒 Locked RTP session SESSION_1 to SSRC 478356044
📍 Initializing RTP session SESSION_2 with actual port 15252 and SSRC 446888240
🔒 Locked RTP session SESSION_2 to SSRC 446888240
✅ Greeting played successfully [Call 1]
✅ Greeting played successfully [Call 2]
[Both calls complete successfully without interference]
```

## Files Modified

1. **`windows_voice_agent.py`**: Fixed RTP session matching logic (lines 2150-2191)
2. **`crm_database_mongodb.py`**: Added `google_api_key` field to User model
3. **`crm_user_management.py`**: Auto-assign API keys to new agents
4. **`crm_superadmin.py`**: Auto-assign API keys to new admins
5. **`crm_api.py`**: Pass owner_id for API key retrieval
6. **`assign_api_keys_to_users.py`**: Script to assign keys to existing users

## Testing

Run `python assign_api_keys_to_users.py` to assign keys to existing users, then:

1. Login as two different users (admin2 + agent4)
2. Make calls simultaneously to different numbers
3. Check logs for:
   - ✅ Different ports assigned (10864 vs 15252)
   - ✅ Different SSRCs (unique identifiers)
   - ✅ Different API keys in use
   - ✅ No "Updating RTP address" messages
   - ✅ Both greetings play successfully
   - ✅ Both calls complete without interference

## Verification Commands

```bash
# 1. Assign API keys to existing users
python assign_api_keys_to_users.py

# 2. Start the server
python windows_voice_agent.py

# 3. Make concurrent calls from two browsers/users
# 4. Watch logs for proper isolation
grep "Initializing RTP session" logs.txt  # Should see two different ports
grep "Updating RTP address" logs.txt      # Should be EMPTY
```

## Summary

The fix ensures that:
- ✅ Each call gets its own RTP port + SSRC combination
- ✅ No port/SSRC stealing between sessions
- ✅ Each user has their own Google API key
- ✅ Complete isolation between concurrent calls
- ✅ No audio mixing or interference
- ✅ Both calls complete successfully

