# AI Sales Services Demo Configuration Added

## Overview

A new professional cold calling demo has been added to the CRM system for selling AI services in Bulgarian. This demo follows best practices for B2B cold calling and includes appointment scheduling functionality.

## What Was Added

### 1. Frontend (CustomCallDialog.js)

#### New Call Objective: `ai_sales_services`

- **Company**: QuantumAI Solutions
- **Caller Name**: Nikolay
- **Product**: AI Асистенти за Продажби и Обслужване на Клиенти (AI Assistants for Sales and Customer Service)
- **Benefits**: 24/7 automated customer service, inquiry qualification, order processing, up to 70% cost reduction, sales increase through quick response, next-generation AI
- **Special Offer**: Free 15-minute online demo and business analysis

#### UI Updates

- Added new dropdown option: "AI Sales Services (Bulgarian)"
- Added info alert explaining the cold call flow with 5 key stages
- Updated all helper texts to include the new demo
- Updated button labels to show "AI Sales" for this objective

### 2. Backend (windows_voice_agent.py)

#### Professional Cold Call Script Structure (in 3 languages)

The AI follows this proven structure:

1. **Introduction & Pattern Interrupt (first 15 seconds)**

   - Professional greeting and company introduction
   - Clear value proposition
   - Pattern interrupt question: "В лош момент ли Ви намирам?" / "Did I catch you at a bad time?"

2. **Value Proposition**

   - Identifies the pain point (lost customers, high support costs)
   - Presents the solution (24/7 AI assistants)
   - Emphasizes key benefits

3. **Qualifying Question**

   - Engages prospect in dialogue
   - Identifies their specific challenges
   - Opens conversation naturally

4. **Appointment Scheduling** ⭐

   - Suggests 15-minute demo
   - Provides specific time options (Tuesday afternoon, Wednesday morning)
   - **NEW**: AI remembers agreed appointment times
   - Format examples: "вторник 16:00", "сряда 10:00", "петък следобед"

5. **Objection Handling**
   - "Send me an email" → Qualify interest first, then schedule
   - "Not interested" → Understand why, offer value
   - "Already have a chatbot" → Differentiate with intelligence
   - "No time" → Acknowledge and propose flexible scheduling

### 3. Appointment Tracking Feature ⭐

**Key Innovation**: When the AI successfully schedules an appointment, it is instructed to:

- Remember the day and time mentioned
- Confirm it at the end of the conversation
- The appointment details will be captured in the conversation transcript

The system is designed to extract appointment information like:

- "Tuesday 16:00" (вторник 16:00)
- "Wednesday 10am" (сряда 10:00)
- "Friday afternoon" (петък следобед)

This information will be available in:

- Session transcripts
- Call analysis results
- Follow-up notes field

## How to Use

1. **Select a lead** from your CRM
2. Click **"Make Call"**
3. In the call objective dropdown, select **"AI Sales Services (Bulgarian)"**
4. The demo configuration will automatically populate:
   - Company: QuantumAI Solutions
   - Caller: Nikolay
   - Product details and benefits
   - Special offer
5. Optionally customize any fields
6. **Generate greeting** (optional but recommended)
7. Click **"Make AI Sales Call"**

## Expected Behavior

The AI will:

- Follow the professional cold calling script structure
- Speak in Bulgarian (or English if phone number is non-Bulgarian)
- Handle objections professionally using proven techniques
- Focus on scheduling a 15-minute demo/meeting
- Remember and confirm any agreed appointment times
- Maintain a professional, consultative (not pushy) tone

## Example Flow

```
AI: "Добър ден, [Name]. Казвам се Nikolay и се обаждам от QuantumAI Solutions.
     Причината за обаждането ми е, че помагаме на компании във Вашия бранш да
     автоматизират обслужването на клиенти и да увеличат продажбите си чрез
     изкуствен интелект. В лош момент ли Ви намирам?"

Customer: "Имам минута..."

AI: "Благодаря Ви. Накратко, ние забелязваме, че много бизнеси губят потенциални
     клиенти или плащат твърде високи разходи за екипи по поддръжка...
     [continues with value proposition]"

AI: "Как се справяте с потока от клиентски запитвания във Вашата компания в момента?"

Customer: [Responds with challenges]

AI: "Разбирам. Предлагам един кратък, 15-минутен разговор следващата седмица...
     Как сте с времето във вторник следобед или в сряда сутрин?"

Customer: "Вторник следобед може."

AI: "Чудесно! Вторник следобед в 16:00 часа Ви подхожда ли?"

Customer: "Да, добре."

AI: "Отлично! Значи насрочваме за вторник в 16:00. [Confirms and closes]"
```

The appointment "Tuesday 16:00" will be captured in the session notes.

## Technical Details

### Configuration Object

```javascript
{
  company_name: "QuantumAI Solutions",
  caller_name: "Nikolay",
  product_name: "AI Асистенти за Продажби и Обслужване на Клиенти",
  main_benefits: "24/7 автоматизирано обслужване на клиенти, квалифициране на запитвания, обработка на поръчки, намаляване на разходите с до 70%, увеличаване на продажбите чрез бърза реакция, AI от следващо поколение",
  special_offer: "Безплатна 15-минутна онлайн демонстрация и бизнес анализ...",
  objection_strategy: "educational"
}
```

### System Prompt Features (Bulgarian)

- Comprehensive cold calling script embedded in system instructions
- Pattern interrupt techniques
- Qualification questions
- Objection handling responses
- Appointment scheduling logic
- Memory instructions for capturing appointment times

## Benefits of This Demo

1. **Professional Cold Calling**: Based on proven B2B sales techniques
2. **Objection Handling**: Pre-programmed responses to common objections
3. **Appointment Focus**: Goal is to schedule demos, not push for immediate sale
4. **Natural Flow**: Conversational rather than scripted
5. **Appointment Tracking**: Remembers and confirms meeting times
6. **Educational Approach**: Focuses on value and benefits
7. **Multi-language**: Works in Bulgarian, English, and other languages

## Files Modified

- `crm-frontend/src/components/CustomCallDialog.js` - Added new demo configuration and UI elements
- `windows_voice_agent.py` - Added AI sales services cold calling scripts in 3 language templates

## Notes

- The demo is optimized for Bulgarian business culture and language
- The script follows AIDA (Attention, Interest, Desire, Action) principles
- Focuses on B2B sales (business owners, managers, sales/support leaders)
- Appointment times are mentioned in natural language format
- Future enhancement: Automatic extraction and structured storage of appointment data in database

---

**Status**: ✅ Fully Implemented and Ready for Testing
**Date**: October 28, 2025
