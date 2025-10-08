import sqlite3

conn = sqlite3.connect('voice_agent_crm.db')
cursor = conn.cursor()

# Check if session exists
session_id = '417252eb-ca15-4ca7-b518-a5e053ad59d1'
cursor.execute('SELECT id, session_id, owner_id, campaign_id FROM call_sessions WHERE session_id = ?', (session_id,))
result = cursor.fetchone()

print(f"\n🔍 Checking session: {session_id}")
if result:
    print(f"✅ Session EXISTS in database:")
    print(f"   ID: {result[0]}")
    print(f"   Session ID: {result[1]}")
    print(f"   Owner ID: {result[2]}")
    print(f"   Campaign ID: {result[3]}")
else:
    print(f"❌ Session NOT FOUND in database")

# Check all sessions
cursor.execute('SELECT COUNT(*) FROM call_sessions')
total = cursor.fetchone()[0]
print(f"\n📊 Total call sessions: {total}")

cursor.execute('SELECT COUNT(*) FROM call_sessions WHERE owner_id IS NULL')
null_owner = cursor.fetchone()[0]
print(f"📊 Sessions with NULL owner_id: {null_owner}")

cursor.execute('SELECT COUNT(*) FROM call_sessions WHERE owner_id IS NOT NULL')
with_owner = cursor.fetchone()[0]
print(f"📊 Sessions with owner_id: {with_owner}")

# Show recent sessions
print(f"\n📋 Recent sessions:")
cursor.execute('SELECT id, session_id, owner_id FROM call_sessions ORDER BY id DESC LIMIT 5')
for row in cursor.fetchall():
    print(f"   ID {row[0]}: {row[1][:20]}... - owner_id: {row[2]}")

conn.close()

