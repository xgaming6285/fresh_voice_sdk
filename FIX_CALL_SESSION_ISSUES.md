# 🔧 Call Session Issues - Analysis & Fixes

## 🐛 Issues Identified from Logs

### 1. **Main Issue: Session Not in Database (404)**

```
GET /api/crm/sessions/417252eb-ca15-4ca7-b518-a5e053ad59d1 HTTP/1.1" 404 Not Found
```

**Root Cause**: Call session was processed by voice agent but never saved to `call_sessions` table.

**Why This Happens**:

- Call was made directly (not through CRM campaign API)
- OR database commit failed silently
- OR there's a mismatch between voice agent session_id and CRM session_id

### 2. **Async Cleanup Errors**

```
ERROR:asyncio:Task was destroyed but it is pending!
task: <Task pending name='Task-207' coro=<Connection.keepalive()...
```

**Root Cause**: WebSocket keepalive task not properly cancelled before event loop closes.

### 3. **Unicode Encoding Error**

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 54
```

**Root Cause**: Windows uses cp1252 encoding, but transcription subprocess outputs UTF-8.

### 4. **Connection Reset Errors**

```
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed
```

**Root Cause**: Gemini WebSocket connections not gracefully closed after call ends.

### 5. **Multiple BYE Messages**

```
WARNING:windows_voice_agent:⚠️ BYE received from unknown address
```

**Root Cause**: SIP server sending multiple BYE messages, but session already cleaned up.

## 🛠️ Required Fixes

### Fix 1: Frontend - Handle 404 Gracefully

**File**: `crm-frontend/src/pages/SessionDetail.js` (or wherever session details are fetched)

**Problem**: Frontend crashes when session returns 404

**Solution**: Add error handling:

```javascript
try {
  const session = await api.get(`/api/crm/sessions/${sessionId}`);
  setSessionData(session.data);
} catch (error) {
  if (error.response?.status === 404) {
    setError(
      "Call session not found in database. The call may have been made outside the CRM system."
    );
    // Still try to fetch recordings/transcripts from file system
    await loadRecordingsFromFileSystem(sessionId);
  } else {
    setError("Failed to load session data");
  }
}
```

### Fix 2: Backend - Default Owner for Direct Calls

**File**: `crm_api.py`

**Problem**: Calls made directly don't have owner_id

**Solution**: Modify `/api/crm/sessions/{session_id}` to handle missing database entries:

```python
@crm_router.get("/sessions/{session_id}", response_model=CallSessionResponse)
async def get_call_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific call session"""
    try:
        session = get_session()
        call_session = session.query(CallSession).filter(
            CallSession.session_id == session_id
        ).first()

        if not call_session:
            # Check if recording exists on disk but not in database
            import os
            recording_dir = f"sessions/{session_id}"
            if os.path.exists(recording_dir):
                # Create database entry for orphaned recording
                call_session = CallSession(
                    session_id=session_id,
                    owner_id=current_user.id,  # Assign to current user
                    status=CallStatus.COMPLETED,
                    started_at=datetime.utcnow()
                )
                session.add(call_session)
                session.commit()
            else:
                raise HTTPException(status_code=404, detail="Call session not found")

        # Check access rights...
        # (rest of the existing code)
```

### Fix 3: Subprocess Encoding

**File**: `windows_voice_agent.py`

**Problem**: subprocess.Popen doesn't specify encoding for Windows

**Current**:

```python
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
```

**Fixed**:

```python
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    encoding='utf-8',  # Force UTF-8 encoding
    errors='replace'   # Replace invalid characters
)
```

### Fix 4: Async Cleanup

**File**: `windows_voice_agent.py`

**Problem**: WebSocket keepalive tasks not properly cancelled

**Solution**: Add proper cleanup in session cleanup:

```python
async def cleanup_session(session_id):
    """Clean up session resources"""
    try:
        # Cancel any pending tasks
        if hasattr(session, 'keepalive_task'):
            session.keepalive_task.cancel()
            try:
                await session.keepalive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connections
        if hasattr(session, 'ws_connection'):
            await session.ws_connection.close()

        # Remove from active sessions
        active_sessions.pop(session_id, None)

    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
```

### Fix 5: Prevent Multiple BYE Responses

**File**: `windows_voice_agent.py`

**Solution**: Track BYE messages per session:

```python
# Add to session tracking
bye_received = {}

def handle_bye(address, session_id):
    # Check if already processed
    if session_id in bye_received:
        logger.debug(f"Duplicate BYE for {session_id}, ignoring")
        return

    bye_received[session_id] = True
    # Process BYE...
    # Clean up after delay
    asyncio.create_task(cleanup_bye_tracking(session_id))

async def cleanup_bye_tracking(session_id):
    await asyncio.sleep(30)  # Keep for 30 seconds
    bye_received.pop(session_id, None)
```

## 🎯 Immediate Actions

### 1. **Quick Fix - Clear Old Data**

```bash
# The database is already clean (0 sessions)
# But there are recording files on disk without database entries
```

### 2. **Restart Backend with Encoding Fix**

Add to subprocess calls:

```python
# In windows_voice_agent.py, find subprocess.Popen calls and add:
encoding='utf-8',
errors='replace'
```

### 3. **Frontend Error Handling**

Update SessionDetail page to handle 404s gracefully and show appropriate messages.

## 📝 Why This Happened

1. **Direct Calls**: If you called the number directly (not through a campaign), the voice agent handled it but didn't create a CRM database entry.

2. **Campaign Calls**: If started through campaign, the entry should have been created in `start_campaign` endpoint, but might have failed.

3. **Session ID Mismatch**: Voice agent uses one session_id format, CRM might expect another.

## ✅ Testing After Fixes

1. **Test Campaign Call**:

   - Start campaign through CRM
   - Make call
   - Check database: `python check_sessions.py`
   - Verify owner_id is set
   - View in UI - should work

2. **Test Direct Call** (if applicable):

   - Call number directly
   - Check database - should auto-create entry when viewed
   - UI should handle gracefully

3. **Test Transcription**:
   - Make call
   - Create transcript
   - Should not see Unicode errors

## 🚀 Long-term Solution

**Option A**: Always create database entries for ALL calls (even direct ones)

- Modify voice agent to write to CRM database on call start
- Requires coupling voice agent with CRM database

**Option B**: Lazy database creation

- Create database entry when user first views recording
- Simpler, but recordings might be "orphaned" until viewed

**Recommendation**: Option B with proper error handling in frontend.

## 📊 Current Status

- ✅ Database clean (no orphaned sessions)
- ❌ Recording files exist without database entries
- ❌ Frontend crashes on 404
- ❌ Async cleanup errors
- ❌ Encoding errors

All fixable with the solutions above!
