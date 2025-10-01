# Greeting Playback Fix

## Issue

When initiating calls from the CRM, the greeting was not playing correctly:

- Sometimes the greeting played too early during the ringing phase
- Sometimes the greeting didn't play at all
- The timing was inconsistent, especially for long ringing periods

## Root Cause

1. The greeting was playing with only a 0.5-second delay after call establishment
2. There were no proper checks to ensure the RTP session was fully ready
3. The system didn't wait for the call to be properly answered before playing the greeting

## Solution Implemented

### 1. Increased Delay

- Changed the delay from 0.5 seconds to 3 seconds as requested
- This gives the RTP connection more time to stabilize

### 2. Added RTP Session Readiness Checks

- Added checks to ensure `output_processing` is active
- Added verification that the session status is "active"
- Will wait up to 5 seconds for the session to be ready

### 3. Added Call State Verification

- Check if the call is still active before playing greeting
- Proper error handling if call ends before greeting can be played

### 4. Enhanced 180 Ringing Handling

- Added proper handling of 180 Ringing responses for outbound calls
- Clear logging to show when call is ringing vs answered
- Greeting only plays after 200 OK (call answered) is received

### 5. Improved Logging

- Added detailed logging of RTP session state
- Better error messages when greeting fails to play
- Clear indication of call status during greeting playback

## Code Changes

### In `windows_voice_agent.py`:

1. **Incoming Calls** (lines 2421-2453):

   - Wait 3 seconds before playing greeting
   - Check RTP session readiness
   - Verify call is still active

2. **Outbound Calls** (lines 3267-3299):

   - Same improvements as incoming calls
   - Specific handling for outbound call scenarios

3. **RTP Session** (lines 2042-2055):

   - Added check for `output_processing` flag
   - Better logging of session state

4. **SIP Response Handler** (lines 3374-3393):
   - Added handling for 180 Ringing responses
   - Clear indication when outbound calls are ringing

## Testing Instructions

1. **Test Incoming Calls**:

   - Call the voice agent number
   - Verify greeting plays 1 second after answering
   - Check logs for proper timing

2. **Test Outbound Calls from CRM**:

   - Navigate to Leads page
   - Click call button on a lead
   - Verify greeting plays 1 second after the recipient answers
   - Check that greeting doesn't play during ringing

3. **Test Edge Cases**:
   - Long ringing periods (let it ring for 10+ seconds)
   - Quick hangups (hang up before greeting plays)
   - Network delays

## Expected Behavior

1. Call is initiated (incoming or outbound)
2. For outbound: "180 Ringing" logged while waiting for answer
3. Call is answered (200 OK received)
4. 3-second delay
5. RTP session readiness check (up to 5 seconds)
6. Greeting plays
7. Natural conversation begins

## Log Messages to Look For

Success case:

```
🔔 Outbound call to +1234567890 is ringing...
📞 Waiting for answer - greeting will play 3 seconds after call is answered
✅ Outbound call to +1234567890 answered successfully!
🎯 Outbound voice session xxx is now active and ready
🎵 Playing greeting to called party...
📞 Outbound call status: active
✅ Outbound greeting played successfully (2.5s)
🎤 Waiting for called party to speak - AI will respond naturally
```

Error cases:

```
⚠️ RTP output processing not ready, cannot play greeting
⚠️ Greeting file not found or failed to play
⚠️ Outbound call ended before greeting could be played
```
