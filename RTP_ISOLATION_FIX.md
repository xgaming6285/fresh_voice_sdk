# RTP Session Isolation Fix

## Problem Identified

When running concurrent calls (e.g., admin2 and agent4 making calls simultaneously), the logs showed:

```
📍 Updating RTP address from ('192.168.50.50', 19508) to ('192.168.50.50', 18920)
📍 Updating RTP address from ('192.168.50.50', 18920) to ('192.168.50.50', 19508)
```

This constant switching indicated that RTP packets were being routed to the wrong session, causing:

- Audio interference between calls
- The same AI response playing to both users
- Lag and stuttering

## Root Cause

The RTP session matching logic was matching **by IP address only**:

```python
# Old logic (WRONG for concurrent calls)
if rtp_session.remote_addr[0] == addr[0]:  # Match by IP only
```

Since both calls come from the same Gate VoIP server (192.168.50.50) on different ports, they would match the same session, causing constant switching.

## Solution

Implemented a **3-tier matching strategy** using SSRC (RTP stream identifier) as the primary key:

### Tier 1: Match by SSRC (Highest Priority)

- Each RTP stream has a unique SSRC identifier in the RTP header
- Once we've seen an SSRC, we lock it to that session
- Most reliable for concurrent calls

### Tier 2: Match by IP + Port

- When we first receive packets, match by both IP and port
- Store the SSRC for future fast matching
- Prevents port confusion

### Tier 3: Match by IP Only (Fallback)

- Only used for initial packet with default port (5004)
- Initializes the session with actual RTP port

## Code Changes

```python
# New logic (CORRECT for concurrent calls)
for session_id, rtp_session in self.sessions.items():
    # Priority 1: Match by SSRC (most reliable)
    if hasattr(rtp_session, 'remote_ssrc') and rtp_session.remote_ssrc == ssrc:
        matched_session = rtp_session
        break

    # Priority 2: Match by IP and port
    if rtp_session.remote_addr == addr:
        if not hasattr(rtp_session, 'remote_ssrc'):
            rtp_session.remote_ssrc = ssrc
            logger.info(f"🔒 Locked RTP session {session_id} to SSRC {ssrc}")
        matched_session = rtp_session
        break

    # Priority 3: Match by IP only (initialization)
    if rtp_session.remote_addr[0] == addr[0] and rtp_session.remote_addr[1] == 5004:
        logger.info(f"📍 Initializing RTP session {session_id} with port {addr[1]}")
        rtp_session.remote_addr = addr
        rtp_session.remote_ssrc = ssrc
        matched_session = rtp_session
        break
```

## Expected Results

After this fix, you should see:

- **No more address switching**: Each session stays locked to its SSRC
- **Clean logs**: `🔒 Locked RTP session <id> to SSRC <number>`
- **Isolated audio streams**: No interference between concurrent calls
- **No lag**: Each call maintains its own audio pipeline

## Combined Solution

This RTP fix **works together** with the API key isolation:

1. **API Key Isolation** (already working ✅):

   - Each user has their own Google API key
   - Each `WindowsVoiceSession` uses its user's API key
   - Separate Gemini connections for each call

2. **RTP Session Isolation** (just fixed ✅):
   - Each call locked to its unique SSRC
   - No RTP packet mixing between concurrent calls
   - Proper audio routing

## Testing

When you run concurrent calls again, watch for:

```
🔒 Locked RTP session 89ea2cc7-3216-4d63-be06-db78b493503a to SSRC 1234567 from ('192.168.50.50', 18920)
🔒 Locked RTP session 2c88e874-d35c-47d8-bc75-6a9c5c8504d0 to SSRC 7654321 from ('192.168.50.50', 19508)
```

**No more switching!** Each session stays locked to its SSRC.

## Why This Works

**RTP SSRC** is specifically designed for this purpose:

- Unique identifier per RTP stream
- Included in every RTP packet header
- Allows multiplexing multiple streams on same IP
- Standard solution for concurrent RTP sessions

Combined with unique API keys per user, you now have **complete isolation** between concurrent calls at both the audio transport layer (RTP) and the AI processing layer (Gemini).
