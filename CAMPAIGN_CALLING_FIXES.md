# Campaign Calling Fixes - Greeting and AI Behavior

## Issues Fixed

### 1. Greeting Not Playing Error

**Problem**: Campaign calls were failing with error:

```
ERROR:windows_voice_agent:Error playing greeting file: expected str, bytes or os.PathLike object, not NoneType
```

**Root Cause**:

- The `greeting_file` variable could be `None` when the greeting generation API call timed out or failed
- The code was passing `None` directly to `play_greeting_file()` which expects a valid file path

**Solution** (`windows_voice_agent.py`):

```python
# Before (line 4495):
custom_greeting = custom_config.get('greeting_file', 'greeting.wav') if custom_config else 'greeting.wav'
greeting_duration = rtp_session.play_greeting_file(custom_greeting)

# After:
custom_greeting = None
if custom_config and custom_config.get('greeting_file'):
    custom_greeting = custom_config.get('greeting_file')
else:
    custom_greeting = 'greeting.wav'

# Only play greeting if file exists
greeting_duration = 0
if custom_greeting and os.path.exists(custom_greeting):
    greeting_duration = rtp_session.play_greeting_file(custom_greeting)
else:
    logger.warning(f"⚠️ Greeting file not found or not specified: {custom_greeting}")
```

**Benefits**:

- Prevents crash when greeting file is None
- Validates file existence before attempting to play
- Provides clear warning message when greeting is unavailable
- Call continues gracefully even if greeting is missing

---

### 2. AI Talking Too Much (Multiple Questions)

**Problem**:

- AI was asking multiple questions consecutively without waiting for responses
- Particularly noticeable in "AI Real Estate Services (Bulgarian)" objective
- Not professional for B2B sales calls to real estate brokers

**Root Cause**:

- The prompt in `build_prompt_from_config()` was too generic
- No specific guidance on conversation pacing
- Special offer text contained multiple questions that the AI would read all at once

**Solution** (`crm_api.py`):

Added specialized handling for `ai_sales_services` and `companions_services` objectives with structured prompts:

```python
elif call_objective == "ai_sales_services":
    basePrompt = f"You are {caller_name} from {company_name}, a professional B2B sales consultant specializing in AI automation solutions for real estate agencies. "
    basePrompt += f"Your product is {product_name}. "
    basePrompt += "\n\n🎯 CALL STRUCTURE - FOLLOW STRICTLY:\n"
    basePrompt += "1) After the greeting, introduce yourself briefly and state your purpose in ONE sentence\n"
    basePrompt += "2) Ask ONE qualifying question: 'Do you currently handle property inquiries manually?'\n"
    basePrompt += "3) WAIT for their response. Do NOT continue talking.\n"
    basePrompt += "4) Based on their answer, present ONE key benefit from these options:\n"
    basePrompt += f"   - {main_benefits}\n"
    basePrompt += "5) Offer the demo: 'We can show you in a quick 15-minute demo how this works.'\n"
    basePrompt += "6) If interested, suggest a specific time: 'Would tomorrow afternoon or Friday morning work better for you?'\n"
    basePrompt += "7) If they have concerns, address them one at a time.\n"
    basePrompt += "\n\n⚠️ CRITICAL RULES:\n"
    basePrompt += "- ASK ONLY ONE QUESTION AT A TIME\n"
    basePrompt += "- WAIT for their response before speaking again\n"
    basePrompt += "- Keep responses under 2-3 sentences\n"
    basePrompt += "- Be professional and consultative, NOT pushy\n"
    basePrompt += "- Listen actively to their needs\n"
    basePrompt += "- If they're not interested, thank them politely and end the call\n"
```

Also added general improvement for all other call types:

```python
basePrompt += "\n\n⚠️ IMPORTANT: ASK ONE QUESTION AT A TIME and WAIT for responses. Keep your statements brief and conversational.\n\n"
```

**Benefits**:

- Professional, consultative B2B sales approach
- Structured conversation flow prevents rambling
- Clear instructions to wait for responses
- Brief, focused responses (2-3 sentences max)
- Better qualification and listening
- Graceful exit if prospect not interested

---

## Testing Recommendations

1. **Test Greeting Handling**:

   - Start a campaign when greeting generation is slow/failing
   - Verify call proceeds without crashing
   - Check logs for appropriate warning messages

2. **Test AI Behavior**:

   - Make test calls with "AI Real Estate Services" objective
   - Verify AI asks one question at a time
   - Confirm AI waits for responses before continuing
   - Ensure responses are brief (2-3 sentences)
   - Check that AI is professional and consultative

3. **Test Campaign Flow**:
   - Create campaign with multiple leads
   - Start with "AI Real Estate Services" configuration
   - Verify calls are made sequentially
   - Confirm greeting plays for each lead (when available)
   - Check that custom prompt is applied consistently

---

## Files Modified

1. **windows_voice_agent.py** (lines 4494-4506)

   - Added null safety check for greeting_file
   - Added file existence validation
   - Improved error handling and logging

2. **crm_api.py** (lines 1178-1296)
   - Added specialized prompt structure for `ai_sales_services`
   - Added specialized prompt structure for `companions_services`
   - Added conversation pacing rules for all objectives
   - Structured B2B sales approach with step-by-step flow
   - Added critical rules section emphasizing one question at a time

---

## Configuration Example

When using the "AI Real Estate Services (Bulgarian)" configuration:

```javascript
{
  company_name: "PropTech AI Solutions",
  caller_name: "Viktor",
  product_name: "AI Асистент за Автоматизация на Недвижими Имоти",
  call_objective: "ai_sales_services",
  main_benefits: "автоматично отговаряне на клиентски запитвания 24/7, интелигентно квалифициране на купувачи, автоматизирано насрочване на огледи, AI генериране на описания на имоти, виртуални асистенти за първоначален контакт, освобождаване на време за продажби",
  special_offer: "Безплатна 15-минутна демонстрация...",
  objection_strategy: "educational",
  voice_name: "Puck"
}
```

The AI will now:

1. Greet briefly (greeting file played if available)
2. Introduce purpose in one sentence
3. Ask ONE qualifying question
4. WAIT for response
5. Present ONE benefit based on their answer
6. Offer demo with specific time options
7. Handle objections one at a time

---

## Notes

- The greeting generation timeout was already increased to 30 seconds in previous changes
- If greeting generation fails, the call will continue without a greeting (graceful degradation)
- The structured prompt approach significantly improves call quality and professionalism
- The "one question at a time" rule is now explicitly enforced in the prompt
- All call objectives now have improved conversation pacing guidelines
