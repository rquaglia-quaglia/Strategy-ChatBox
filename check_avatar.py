import requests
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("SYNTHESIA_API_TOKEN")

if not token:
    print("❌ No token found. Check your .env file.")
    exit()

headers = {
    "Authorization": f"Bearer {token}"
}

print("🔎 Fetching avatar list from Synthesia API...")
response = requests.get("https://api.synthesia.io/v2/avatars", headers=headers)
print(f"Status: {response.status_code}")
print(response.json())
