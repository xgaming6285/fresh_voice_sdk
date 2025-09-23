# Gate VoIP Configuration Guide for Voice Agent

## Overview

This guide explains how to configure Gate VoIP to support both INCOMING and OUTGOING calls with your AI voice agent.

## Current Configuration Analysis

Based on your screenshots, you have:

- ✅ **Extension 200**: `voice-agent-200` (working for incoming calls)
- ✅ **Trunk "gsm2"**: Dynamic trunk (for outgoing calls to external numbers)
- ✅ **Trunk "voice-agent"**: Static trunk (for receiving calls from external sources)
- ✅ **Incoming Route**: `voice-agent-incoming` → forwards to Extension 200
- ❌ **Outgoing Route**: Currently misconfigured to use "voice-agent" trunk instead of "gsm2"

## Required Changes

### 1. Keep Current Incoming Configuration (✅ Already Working)

```
External Call → GSM2 Trunk → Gate VoIP → voice-agent trunk → Extension 200 (AI Bot)
```

### 2. Fix Outgoing Route Configuration (❌ Needs Fix)

In your Gate VoIP web interface:

1. **Go to: PBX Settings → Outgoing Routes**
2. **Find the route that handles outbound calls**
3. **Change the Trunk selection from "voice-agent" to "gsm2"**

The correct outgoing configuration should be:

```
Extension 200 (AI Bot) → Gate VoIP → GSM2 Trunk → External Call
```

### 3. Verify Extension 200 Permissions

Make sure Extension 200 has outbound calling permissions:

1. **Go to: PBX Settings → Extensions → 200**
2. **Check "Outgoing context" is set to allow external calls**
3. **Verify the extension can use the GSM2 trunk**

## Detailed Configuration Steps

### Step 1: Update Outgoing Route

1. Open Gate VoIP web interface: http://192.168.50.50
2. Navigate to: **PBX Settings → Outgoing Routes**
3. Edit the outgoing route (shown in your last screenshot)
4. Change **Trunk** from "voice-agent" to "gsm2"
5. Save the configuration

### Step 2: Verify Dialplan Context

Make sure the outgoing route pattern matches the numbers you want to call:

- For Bulgarian mobile numbers: `+359*` or `359*`
- For international calls: `+*` or `00*`

### Step 3: Test Extension Registration

After starting the voice agent, you should see in the logs:

```
✅ Successfully registered as Extension 200!
📞 Voice agent can now make outbound calls
```

## Call Flow Diagrams

### Incoming Calls (✅ Working)

```
External Number → GSM Gateway → GSM2 Trunk → Gate VoIP → voice-agent trunk → Extension 200 (AI Bot)
```

### Outgoing Calls (🔧 After Fix)

```
AI Bot (Extension 200) → Gate VoIP → GSM2 Trunk → GSM Gateway → External Number
```

## Troubleshooting

### If Extension Registration Fails:

1. Check Extension 200 password in `asterisk_config.json`
2. Verify Extension 200 exists in Gate VoIP
3. Check network connectivity between voice agent and Gate VoIP

### If Outbound Calls Fail:

1. Verify GSM2 trunk is online and registered
2. Check outgoing route pattern matches the dialed number
3. Verify Extension 200 has outbound calling permissions
4. Check GSM gateway has sufficient credit/connection

### If Incoming Calls Stop Working:

1. Verify voice-agent trunk is still configured
2. Check incoming route still points to Extension 200
3. Ensure Extension 200 is registered and available

## Testing

### Test Incoming Calls:

1. Call your configured phone number: `+359898995151`
2. Should hear greeting and be able to talk to AI

### Test Outbound Calls:

1. Use the API endpoint: `POST /api/make_call`
2. Body: `{"phone_number": "+359XXXXXXXXX"}`
3. Should initiate call through GSM2 trunk

## Configuration Summary

After applying these changes, your setup will be:

**For Incoming:**

- External → GSM2 → Gate VoIP → voice-agent trunk → Extension 200 (Bot)

**For Outgoing:**

- Extension 200 (Bot) → Gate VoIP → GSM2 trunk → External

This gives you a complete bidirectional AI voice agent that can both receive and make calls!
