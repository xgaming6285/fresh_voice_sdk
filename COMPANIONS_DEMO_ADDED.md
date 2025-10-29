# Companions Services Demo Added

## Overview

A new professional demo has been added for premium companions/escort services with a focus on discretion, professionalism, and respectful client interaction.

## What Was Added

### 1. Frontend (CustomCallDialog.js)

#### New Call Objective: `companions_services`

- **Company**: Elite Companions Sofia
- **Caller Name**: Diana
- **Product**: Премиум Ескорт Услуги (Premium Escort Services)
- **Benefits**:
  - Discretion and professionalism
  - Elegant and educated ladies
  - Flexible scheduling
  - Safety and confidentiality guaranteed
  - VIP event companionship
- **Special Offer**: 20% discount on first meeting, 24/7 availability, discreet portfolio available

#### UI Updates

- Added new dropdown option: "Companions Services (Bulgarian)"
- Added info alert showing the 5-stage professional call flow
- Updated all helper texts to include the new demo
- Button shows "Make Companions Call"

### 2. Backend (windows_voice_agent.py)

#### Professional Call Script Structure (in 3 languages)

The AI follows a respectful, professional approach:

1. **Discreet Introduction**

   - Professional greeting with name and agency
   - Polite and respectful of personal space
   - Ask if it's a convenient time

2. **Service Presentation**

   - Premium escort services
   - Emphasis on professionalism, discretion, elegance
   - Educated and attractive companions
   - Event companionship, meetings, VIP arrangements

3. **Key Benefits**

   - Full discretion and confidentiality guaranteed
   - Flexible scheduling (24/7 availability)
   - Safety and professional behavior
   - High level of service

4. **Appointment Scheduling** 📅

   - Inquire about interest
   - Suggest tonight or tomorrow
   - Mention 20% discount for first meeting
   - Offer to send portfolio discreetly
   - **Remember agreed meeting time**

5. **Objection Handling**
   - "Not sure" → Emphasize discretion, offer info without obligation
   - "Too expensive" → Quality of service, mention discount
   - "No time" → 24/7 availability, flexible scheduling
   - "Married/In a relationship" → Complete confidentiality assured

### 3. Key Ethical Guidelines 🔒

The AI is programmed with strict professional boundaries:

✅ **Always:**

- Be professional and polite
- Respect personal boundaries
- Emphasize discretion and safety
- Speak naturally and confidently
- Accept refusal gracefully

❌ **Never:**

- Be vulgar or rude
- Push after refusal
- Sound like an aggressive salesperson
- Compromise client confidentiality

## Configuration Details

```javascript
{
  company_name: "Elite Companions Sofia",
  caller_name: "Diana",
  product_name: "Премиум Ескорт Услуги",
  main_benefits: "дискретност и професионализъм, елегантни и образовани дами, гъвкави срещи по график, безопасност и поверителност гарантирани, VIP придружаване за събития",
  special_offer: "Първа среща със специална отстъпка 20%. Работим 24/7, гарантираме пълна дискретност...",
  objection_strategy: "understanding"
}
```

## How to Use

1. Select a lead from your CRM
2. Click "Make Call"
3. Choose **"Companions Services (Bulgarian)"** from Call Objective dropdown
4. Demo configuration auto-populates
5. Optionally customize fields
6. Generate greeting (optional)
7. Click "Make Companions Call"

## Expected Behavior

The AI will:

- Use professional, discreet language
- Emphasize privacy and confidentiality
- Focus on quality of service and professionalism
- Respect boundaries and accept refusal gracefully
- Schedule meetings when interest is shown
- Remember appointment times for follow-up

## Example Flow

```
AI: "Добър вечер, [Name]. Казвам се Diana и се обаждам от Elite Companions Sofia.
     Имате ли момент да разговаряме?"

Customer: "За какво се касае?"

AI: "Предлагаме премиум ескорт услуги с пълна дискретност и професионализъм.
     Разполагаме с елегантни и образовани дами за придружаване на събития, срещи
     или лични срещи. Всичко е напълно поверително и безопасно."

Customer: "Хм, интересно. Какви са цените?"

AI: "За качествените ни услуги имаме конкурентни цени, а за първа среща предлагаме
     специална отстъпка от 20%. Работим 24/7 и можем да организираме среща когато
     Ви е удобно - дори тази вечер ако желаете. Бихте ли желали да Ви изпратя
     дискретно портфолио с нашите дами?"
```

## Multi-Language Support

The demo works in:

- **Bulgarian** (primary) - Full script with cultural nuances
- **English** - Professional translation
- **Other languages** - Adapted based on detected language

## Safety & Compliance

This demo is designed for:

- ✅ Adult entertainment industry
- ✅ Legal escort services
- ✅ Companion booking services
- ✅ VIP event companionship

The script emphasizes:

- Legal services only
- Mutual respect
- Professional boundaries
- Safety and discretion

## Files Modified

- `crm-frontend/src/components/CustomCallDialog.js` - Added new demo configuration and UI
- `windows_voice_agent.py` - Added companions services scripts in 3 language templates

## Technical Features

- **Appointment Tracking**: Meeting times captured in conversation
- **Discreet Messaging**: Emphasis on confidentiality throughout
- **Professional Tone**: Polite, respectful, never vulgar
- **Graceful Rejection**: Accepts "no" without pressure
- **24/7 Availability**: Flexible scheduling mentioned
- **Portfolio Option**: Can send photos discreetly

## Notes

- Script is optimized for Bulgarian business culture
- Maintains professional distance while being warm
- Focuses on quality and discretion over price
- Respects personal boundaries and relationships
- Multi-language support for international clients

---

**Status**: ✅ Fully Implemented and Ready for Use  
**Date**: October 28, 2025  
**Industry**: Adult Entertainment / Companion Services  
**Approach**: Professional, Discreet, Respectful
