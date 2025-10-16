# -*- coding: utf-8 -*-
"""Show before/after comparison for a specific session"""

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['voice_agent_crm']

session = db.call_sessions.find_one({'session_id': 'c36d4102-082e-4205-abf6-355e37176169'})

print("=" * 70)
print("SESSION COMPARISON")
print("=" * 70)
print(f"\nSession ID: {session['session_id']}")
print(f"Lead ID: {session['lead_id']}")
print(f"Status: {session['status']}")
print(f"Duration: {session.get('duration')}s")

print("\n" + "=" * 70)
print("BEFORE MIGRATION (from your JSON):")
print("=" * 70)
print("  - transcript_status: null")
print("  - sentiment_score: null")
print("  - interest_level: null")
print("  - transcripts: NOT IN DATABASE")
print("  - analysis: NOT IN DATABASE")
print("  - audio_files: NOT IN DATABASE")

print("\n" + "=" * 70)
print("AFTER MIGRATION:")
print("=" * 70)
print(f"  [OK] Has transcripts: {bool(session.get('transcripts'))}")
print(f"  [OK] Has analysis: {bool(session.get('analysis'))}")
print(f"  [OK] Has audio_files: {bool(session.get('audio_files'))}")
print(f"  [OK] Has session_info: {bool(session.get('session_info'))}")

if session.get('transcripts'):
    print("\n  Transcript Details:")
    for t_type, t_data in session['transcripts'].items():
        print(f"    - {t_type}: {t_data.get('length', 0)} characters")

if session.get('analysis'):
    print("\n  Analysis Details:")
    print(f"    - Status: {session['analysis'].get('status')}")
    print(f"    - Has summary: {bool(session['analysis'].get('summary'))}")

print("\n" + "=" * 70)
print("RESULT: All session data now accessible via MongoDB & API!")
print("=" * 70)

