# Campaign Calling - Duplicate Gemini Session Fix

## Issue Description

During campaign calling, after approximately 10 calls, the server would crash with the following errors:

```
ERROR:windows_voice_agent:Error receiving from Gemini: sent 1000 (OK); then received 1000 (OK)
ERROR:windows_voice_agent:Error receiving from Gemini: cannot call recv while another coroutine is already running recv or recv_streaming
ERROR:asyncio:Task was destroyed but it is pending!
task: <Task pending name='Task-XXX' coro=<Connection.keepalive()...>>
```

## Root Cause

The issue was caused by **duplicate Gemini voice session initialization**:

1. When an outbound call is made, a Gemini session is initialized in `_async_main_loop()`
2. A continuous response receiver task is started to listen for Gemini responses
3. After the greeting is played (~11 seconds), something was triggering a **second** call to `initialize_voice_session()`
4. This created a second WebSocket connection and started a second continuous receiver
5. Both receivers tried to read from the WebSocket simultaneously, causing the "ConcurrencyError: cannot call recv while another coroutine is already running recv or recv_streaming"

## The Fix

Added a guard check at the beginning of `WindowsVoiceSession.initialize_voice_session()`:

```python
async def initialize_voice_session(self):
    """Initialize the Google Gemini voice session with improved error handling"""
    # Check if session is already initialized
    if self.gemini_session is not None:
        logger.warning(f"⚠️ Voice session already initialized for {self.session_id}, skipping duplicate initialization")
        return True

    # ... rest of initialization code
```

This prevents duplicate initialization by:

- Checking if `gemini_session` is already set before initializing
- If already initialized, returning `True` immediately (successful state)
- If not initialized, proceeding with normal initialization

## Testing

Test the fix by:

1. Starting a campaign with 50+ leads
2. Monitoring the logs for the warning message: "⚠️ Voice session already initialized..."
3. Verifying that the server continues running through all calls without crashing
4. Ensuring no "cannot call recv while another coroutine is already running" errors appear

## Impact

- ✅ Prevents duplicate Gemini WebSocket connections
- ✅ Eliminates WebSocket concurrency errors
- ✅ Allows campaigns to run continuously without crashing
- ✅ No impact on call quality or functionality
- ✅ Minimal performance overhead (simple None check)

## Date

October 30, 2025
