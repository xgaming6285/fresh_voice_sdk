# Greeting Context Fix - Preventing Double Greetings

## Problem

When making calls through the CRM, a greeting was being played twice:

1. First as a static pre-generated audio file
2. Then again by Gemini Live API (which didn't know a greeting was already played)

This resulted in a poor user experience with duplicate introductions.

## Solution

Pass the greeting transcript to Gemini Live API as context, so it knows:

- What was already said to the caller
- Not to repeat the introduction
- To wait for the caller to speak first

## Implementation

### 1. Backend Changes (`windows_voice_agent.py`)

#### API Endpoint Updates

- **`/api/generate_greeting`**: Now returns `greeting_text` in response (line 4754)
- **`/api/make_call`**: Now accepts `greeting_transcript` parameter (line 4775) and adds it to `custom_config` (lines 4809-4811)

#### Voice Configuration Update

Modified `create_voice_config()` function to include greeting context in system instruction:

**For English:**

```python
if greeting_transcript:
    system_text += f"IMPORTANT: You have ALREADY played this greeting to the caller: \"{greeting_transcript}\". DO NOT repeat this greeting. DO NOT introduce yourself again. The caller has already heard your introduction. Wait for the caller to speak first, then respond naturally to what they say. "
```

**For Bulgarian:**

```python
if greeting_transcript:
    system_text += f"ВАЖНО: Вече сте изпратили това приветствие на обаждащия се: \"{greeting_transcript}\". НЕ повтаряйте това приветствие. НЕ се представяйте отново. Обаждащият се вече е чул вашето представяне. Изчакайте обаждащият се първо да говори, след това отговорете естествено на това, което казва. "
```

**For other languages:** Similar instruction in English

### 2. Frontend Changes

#### `CustomCallDialog.js`

- Updated to pass greeting transcript to `onMakeCall` (lines 151-156)
- Extracts `greetingData.transcript` or `greetingData.greeting_text` from API response

#### `Leads.js`

- Updated `handleMakeCustomCall` to accept `greetingTranscript` parameter (line 166)
- Passes it to the API service

#### `api.js`

- Updated `voiceAgentAPI.makeCall` to accept and send `greetingTranscript` (lines 116-129)
- Adds it to the API payload if provided

## Workflow

1. **User initiates call** from CRM with custom configuration
2. **Frontend generates greeting** via `/api/generate_greeting`
3. **API returns** greeting audio file + transcript text
4. **Frontend makes call** via `/api/make_call` with:
   - `greeting_file`: path to audio file to play
   - `greeting_transcript`: text of what was said in the greeting
5. **Backend creates voice config** with greeting context in system instruction
6. **Gemini receives context** and knows not to repeat the greeting
7. **Call flow:**
   - Static greeting plays
   - Gemini waits for caller to speak
   - Gemini responds naturally without re-introducing

## Testing

To test the fix:

1. Create a call from CRM with custom greeting
2. The greeting should play once
3. After greeting, AI should wait for caller response
4. AI should not repeat introduction or greeting

## Benefits

✅ No more double greetings
✅ More natural conversation flow
✅ Better user experience
✅ Greeting context maintained across the entire call
✅ Works for all supported languages (English, Bulgarian, Romanian, Greek, etc.)

## Files Modified

1. `windows_voice_agent.py` - Backend API and voice configuration
2. `crm-frontend/src/components/CustomCallDialog.js` - Call dialog component
3. `crm-frontend/src/pages/Leads.js` - Leads page
4. `crm-frontend/src/services/api.js` - API service layer

## Additional Notes

- The greeting transcript is stored in `custom_config['greeting_transcript']`
- It's passed through the entire call setup flow
- The system instruction is generated dynamically based on language
- For inbound calls (without custom greetings), no greeting context is added
