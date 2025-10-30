"""
Check what recordings the API returns
"""
import requests

response = requests.get("http://localhost:8000/api/recordings")
data = response.json()

print(f"Total recordings: {data['total_recordings']}")
print("\nRecordings:")
print("=" * 100)

for i, rec in enumerate(data['recordings'], 1):
    print(f"\n{i}. Recording ID: {rec['session_id']}")
    print(f"   Time: {rec['start_time']}")
    print(f"   From: {rec['caller_id']} -> To: {rec['called_number']}")
    print(f"   Duration: {rec['duration']}")
    print(f"   Status: {rec['status']}")

