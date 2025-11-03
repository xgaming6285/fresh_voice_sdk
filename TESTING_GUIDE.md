# Quick Testing Guide - Voice Agent Interruption

## What Changed?
Your voice agent now supports **full-duplex** conversation with instant interruption detection. Users can interrupt the AI model at any time, even mid-word.

## Quick Test Scenarios

### Test 1: Simple Interruption
**What to do:**
1. Start a call
2. Let the AI speak for a few seconds
3. Say "but wait" while it's still speaking

**Expected Result:**
- AI stops within 100ms
- You should see in logs: `🔄 User interrupted model - stopping speech immediately!`
- AI hears your complete phrase "but wait" (not just "wait")

### Test 2: Mid-Word Interruption
**What to do:**
1. Let AI give a long explanation
2. Interrupt with a single word like "but" or "hold on"
3. Immediately continue with your thought

**Expected Result:**
- AI catches the interruption word ("but")
- AI processes your complete sentence
- Natural conversation flow

### Test 3: Rapid Exchange
**What to do:**
Have a quick back-and-forth:
- You: "Can you tell me about—"
- AI: "Sure, I can explain—"
- You: "Wait, first—"
- AI: "Yes?"

**Expected Result:**
- No lost words
- Natural turn-taking
- Feels like talking to a human

## What to Look For in Logs

### Good Signs ✅
```
🎙️ Full-duplex mode enabled - user audio always sent to Gemini for instant interruption
🎤 Full-duplex streaming active - all user audio sent to Gemini in real-time
🎙️ Full-duplex mode: User speaking while/after assistant - sending to Gemini for interruption detection
🔄 User interrupted model - stopping speech immediately!
🔇 Cleared X pending audio chunks to stop model speech
🎤 User transcript: but wait  ← Complete phrase captured!
```

### Bad Signs (Shouldn't Happen) ❌
```
🔇 Suppressing incoming audio - assistant is speaking  ← Should NOT appear anymore
🔇 Suppressing incoming audio - grace period  ← Should NOT appear anymore
🎤 Resuming audio processing after echo suppression  ← Should NOT appear anymore
```

## Troubleshooting

### Problem: Model doesn't stop when I speak
**Check:**
1. Are you seeing "User interrupted model" in logs?
2. Is audio being sent? (Check for "📤 Sent X audio packets")
3. Network connectivity stable?

**Solution:**
- Check your microphone is working
- Ensure you're speaking loud enough (try louder)
- Check network latency (high latency = slower interruption)

### Problem: Model hears its own voice (echo)
**Check:**
1. This shouldn't happen - Gemini has built-in echo handling
2. Check your VoIP gateway echo cancellation settings

**Solution:**
- Enable echo cancellation on VoIP gateway
- Use headphones during testing (eliminates acoustic feedback)

### Problem: Lost words at start of interruption
**Check:**
1. Look at "User transcript" logs - is the full phrase there?
2. Check audio chunk size (should be 160 bytes = 10ms)

**Solution:**
- Already optimized! If still happening, reduce chunk size to 80 bytes (5ms)

## Performance Expectations

| Metric | Target | What It Means |
|--------|--------|---------------|
| Interruption detection | <50ms | Time for Gemini to detect you're speaking |
| Model stops speaking | <100ms | Time for AI voice to actually stop |
| Total perceived latency | <150ms | Feels instant to users |
| Audio chunk size | 10ms (160 bytes) | Balance of latency vs efficiency |
| Network usage | ~16KB/s | Negligible for modern networks |

## Comparison: Before vs After

### BEFORE (You mentioned this in your logs)
```
Model: "Разбрах. Много брокери губят потенциални клиенти..."
You: "А" [BLOCKED - not sent]
You: "ми" [SENT after 150ms delay]
Model hears: "ми да, може..."  ← Lost "А"
```

### AFTER (How it should work now)
```
Model: "Разбрах. Много брокери губят потенциални клиенти..."
You: "А" [SENT immediately - 10ms]
[Gemini detects interruption - 30ms]
[Model stops - 50ms]
Model hears: "А ми да, може..."  ← Complete phrase! ✅
```

## Real Conversation Test

Try this exact conversation from your log, but interrupt more aggressively:

**Model:** "Разбрах. Много брокери губят потенциални клиенти, защото не могат да отговорят на всички веднага..."

**You (interrupt mid-sentence):** "Но чакай—" (But wait—)

**Expected:** 
- Model stops immediately after "всички"
- Doesn't finish the sentence
- Acknowledges your interruption: "Да?" or "Слушам"

## Success Criteria

After testing, you should be able to say:
- ✅ I can interrupt the AI at any time
- ✅ The AI stops within a second (feels instant)
- ✅ My first words are heard (not dropped)
- ✅ The conversation feels natural
- ✅ I can have rapid back-and-forth exchanges

## Advanced Testing

If basic testing works well, try these:

### Multiple Interruptions
```
AI: "За да обясня това, трябва първо да разбереш—"
You: "Секунда—"
AI: "Да?"
You: "Всъщност, не—"
AI: "Добре, какво искаш да—"
You: "Чакай—"
```

**Expected:** AI handles all interruptions gracefully

### Background Noise
- Test with music playing
- Test with other people talking nearby
- Test with keyboard typing

**Expected:** Gemini's VAD should filter out most background noise, only respond to direct speech

### Different Languages
From your log, you support Bulgarian. Test interruptions in Bulgarian:
- "Но..." (But...)
- "Чакай..." (Wait...)
- "Секунда..." (Second...)

**Expected:** Works equally well in all languages

## What Changed in the Code

**Key files modified:**
- `windows_voice_agent.py` (4 sections updated)
  - Removed echo suppression blocking
  - Reduced audio chunks from 40ms → 10ms
  - Enhanced interruption response
  - Updated logging

**Lines changed:** ~50 lines
**Behavior change:** Half-duplex → Full-duplex

## Rollback Plan

If something goes wrong, you can rollback by:

1. Restore echo suppression:
```python
if self.audio_output_active or time_since_output < self.echo_suppression_delay:
    return  # Don't process this audio
```

2. Increase chunk size:
```python
min_chunk = 640  # 40ms chunks
```

But this shouldn't be necessary - the new approach is fundamentally better.

## Getting Help

If you see unexpected behavior:

1. **Check logs** - Look for the patterns mentioned above
2. **Share transcript** - Like you did in your original message
3. **Note timing** - When did the interruption happen vs when model stopped?
4. **Test network** - High latency affects interruption detection

## Final Notes

The change is **minimal but fundamental**:
- **Before:** Client blocked audio when model was speaking
- **After:** Client always sends audio, Gemini handles interruptions

This is how Gemini Live API is **designed to work**. You're now using it correctly!

Enjoy natural conversations! 🎉

