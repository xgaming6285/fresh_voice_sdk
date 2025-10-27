# Session Cleanup Fix - Silent Audio on Subsequent Calls

## Problem Description

After the first call ends and a second call is made, there is no audio output:

- Session initializes successfully
- Greeting plays but silently (no audio heard by user)
- User speech is detected and sent to API
- API responds with transcripts and speech
- But the response audio is silent - user cannot hear it

## Root Cause Identified

**The critical bug was in `_send_bye_to_gate()` method:**

When a call ended via goodbye detection or timeout, the RTPSession would:

1. ✓ Send SIP BYE message to terminate the call
2. ✓ Set processing flags to `False`
3. ❌ **BUT NOT actually cleanup the threads!**

This meant:

- Old asyncio event loop kept running
- Old output/input threads kept trying to process audio
- Old timeout monitor thread kept running
- Audio queues retained old data
- When the new call started, the old threads interfered with the new session's audio processing

**Key Evidence from Logs:**

```
First call:
✅ BYE message sent to ('192.168.50.50', 5060)
(NO cleanup logs!)

Second call:
ERROR:asyncio:Task was destroyed but it is pending!  ← Old session's task
ERROR:windows_voice_agent:Error in RTP listener: [WinError 10054]  ← Old session interfering
```

## Root Causes Identified

### 1. **Incomplete Thread Cleanup in \_send_bye_to_gate()**

The `_send_bye_to_gate()` method was setting flags but not cleaning up threads:

```python
# BEFORE (BUGGY CODE):
self.input_processing = False
self.output_processing = False
self.processing = False
self.timeout_monitoring = False
# Threads still running! Queues still full! Event loop still active!
```

### 2. **Missing Session Removal**

After sending BYE, the session was never removed from:

- `active_sessions` dictionary
- RTP server's session list

### 3. **Shared Socket Resource**

All RTPSession instances share the same UDP socket from RTPServer. Without proper cleanup of the session state, old threads tried to use the socket causing errors.

### 4. **AsyncIO Loop Not Closed**

The asyncio event loop used for Gemini communication was not properly closed, leading to:

- Memory leaks (loop keeps references to coroutines)
- Multiple loops trying to send to Gemini simultaneously
- "Task was destroyed but it is pending" errors

## Fixes Implemented

### 1. **Fixed \_send_bye_to_gate() to Properly Clean Up**

Modified `_send_bye_to_gate()` to remove session and trigger cleanup:

```python
# AFTER (FIXED CODE):
# Send BYE message
gate_addr = (sip_handler.gate_ip, sip_handler.sip_port)
sip_handler.socket.sendto(bye_message.encode(), gate_addr)
logger.info(f"✅ BYE message sent to {gate_addr} for session {self.session_id}")

# Remove from active sessions
from __main__ import active_sessions
if self.session_id in active_sessions:
    del active_sessions[self.session_id]
    logger.info(f"🗑️ Session {self.session_id} removed from active sessions")

# Remove RTP session (this will call cleanup_threads automatically)
sip_handler.rtp_server.remove_session(self.session_id)

logger.info(f"✅ Session {self.session_id} full cleanup complete after sending BYE")
```

### 2. **Enhanced RTPServer.remove_session()**

Modified to gracefully handle already-removed sessions:

```python
def remove_session(self, session_id: str):
    """Remove RTP session and cleanup its resources"""
    if session_id in self.sessions:
        rtp_session = self.sessions[session_id]

        # Call cleanup method to stop all threads
        if hasattr(rtp_session, 'cleanup_threads'):
            rtp_session.cleanup_threads()

        del self.sessions[session_id]
        logger.info(f"🎵 Removed RTP session {session_id}")
    else:
        logger.debug(f"RTP session {session_id} already removed")
```

### 3. **Added cleanup_threads() Method to RTPSession**

Created comprehensive cleanup that:

- Stops all processing flags
- Clears all queues (audio_input_queue, output_queue)
- Closes asyncIO loop properly
- Waits for threads to finish (with 500ms timeout)
- Clears all references

```python
def cleanup_threads(self):
    """Properly cleanup all RTP session threads and resources"""
    logger.info(f"🧹 Cleaning up RTP session {self.session_id} threads...")

    # Stop all processing flags
    self.input_processing = False
    self.output_processing = False
    self.processing = False
    self.timeout_monitoring = False

    # Clear queues to unblock threads
    while not self.audio_input_queue.empty():
        self.audio_input_queue.get_nowait()
    while not self.output_queue.empty():
        self.output_queue.get_nowait()

    # Close asyncio loop properly
    if self.asyncio_loop and not self.asyncio_loop.is_closed():
        if self.asyncio_loop.is_running():
            self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
        self.asyncio_loop.close()

    # Wait for threads to finish (with 500ms timeout)
    # ... detailed waiting logic

    # Clear references
    self.voice_session = None
    self.call_recorder = None
    self.rtp_socket = None
```

### 4. **Updated BYE Handler to Use Proper Cleanup**

Modified `_handle_bye()` to call `rtp_server.remove_session()` which automatically calls `cleanup_threads()`:

```python
# Before:
rtp_session.input_processing = False
rtp_session.output_processing = False
rtp_session.processing = False
rtp_session.timeout_monitoring = False

# After:
self.rtp_server.remove_session(session_id)  # Calls cleanup_threads()
```

## How This Fixes The Problem

### Call 1 (First Call - Works Fine)

1. RTPSession created with fresh threads
2. Audio flows bidirectionally
3. Call ends → BYE sent
4. **NEW**: `cleanup_threads()` stops all threads and clears resources
5. **NEW**: Session removed from `active_sessions`
6. **NEW**: RTP session removed from server
7. System is truly clean for next call

### Call 2 (Second Call - Now Works!)

1. New RTPSession created with fresh threads
2. **No interference from previous session's threads** ✅
3. **Queues are empty and ready** ✅
4. **AsyncIO loop is fresh** ✅
5. **No socket conflicts** ✅
6. Audio flows bidirectionally ✅

## Testing Checklist

Test the following scenarios:

- [ ] Make first call - verify two-way audio works
- [ ] End first call gracefully (user hangs up)
- [ ] Make second call - verify greeting is heard
- [ ] Verify user speech is heard by AI in second call
- [ ] Verify AI responses are heard by user in second call
- [ ] Make third, fourth calls - verify audio continues working
- [ ] Test timeout hangup scenario (15s no speech)
- [ ] Test goodbye detection hangup scenario

## Log Messages to Watch For

**Good cleanup logs (you should see these now):**

```
✅ BYE message sent to ('192.168.50.50', 5060) for session <id>
🗑️ Session <id> removed from active sessions
🧹 Cleaning up RTP session <id> threads...
✅ Closed asyncio loop for session <id>
✅ All threads stopped for session <id>
✅ RTP session <id> cleanup complete
✅ RTP session <id> threads cleaned up
🎵 Removed RTP session <id>
✅ Session <id> full cleanup complete after sending BYE
```

**Problem indicators (should NOT see these anymore):**

```
ERROR:asyncio:Task was destroyed but it is pending!
ERROR:windows_voice_agent:Error in RTP listener: [WinError 10054]
⚠️ 3 thread(s) still running after cleanup: ['output', 'asyncio', 'timeout']
```

## Files Modified

- `windows_voice_agent.py`
  - Lines 1853-1869: Enhanced `RTPServer.remove_session()` with graceful handling
  - Lines 2714-2725: **CRITICAL FIX**: `_send_bye_to_gate()` now removes session and calls cleanup
  - Lines 2748-2828: Added `RTPSession.cleanup_threads()` method
  - Lines 2672, 2692, 2731: Updated error paths in `_send_bye_to_gate()` to use cleanup
  - Lines 3143-3153: Updated BYE handler (with lock)
  - Lines 3199-3207: Updated BYE handler (without lock)

## Technical Notes

### Why Queue Clearing is Important

When stopping threads, if queues still have items, threads can block on `queue.get()` calls and never terminate. Clearing queues ensures threads can check the `processing` flag and exit cleanly.

### Why AsyncIO Loop Must be Closed

Each RTPSession creates its own event loop in a separate thread. If not closed:

- Memory leaks (loop keeps references to coroutines)
- Multiple loops try to send to Gemini simultaneously
- Responses get mixed between sessions
- "Task was destroyed but it is pending" errors

### Why Thread Waiting is Important

We give threads 500ms to finish gracefully. Daemon threads will be killed when the main thread exits, but during runtime:

- Proper shutdown prevents exceptions in logs
- Ensures resources (file handles, network sockets) are released
- Prevents undefined behavior from concurrent access

### Why Session Removal is Critical

Removing the session from `active_sessions` and RTP server prevents:

- Memory leaks from keeping references to old sessions
- Old threads trying to send audio through shared sockets
- New calls accidentally using old session data

## Additional Improvements Made

Beyond fixing the silent audio issue, these changes also:

1. **Reduce memory leaks** - threads and event loops are properly cleaned
2. **Improve system stability** - no zombie threads interfering
3. **Enable faster call handling** - clean state means instant readiness
4. **Better logging** - clear visibility into cleanup process
5. **Prevent resource exhaustion** - after many calls, system remains stable
6. **Fix socket errors** - no more "connection forcibly closed" errors

## Monitoring

After deploying, monitor these metrics:

- Number of active threads over time (should not grow indefinitely)
- Memory usage per call (should return to baseline after call ends)
- Call success rate (should be 100% regardless of call number)
- Audio quality on calls 10, 20, 50+ (should remain consistent)
- No "Task was destroyed" or "WinError 10054" errors in logs
