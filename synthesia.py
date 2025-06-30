import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

def generate_avatar_video(text):
    token = os.getenv("SYNTHESIA_API_TOKEN")
    avatar_id = os.getenv("SYNTHESIA_AVATAR_ID")

    # Debugging prints (inside function!)
    print(f"✅ Using Synthesia Token: {token}")
    print(f"🧑‍🚀 Using Avatar ID: {avatar_id}")

    if not token or not avatar_id:
        print("❌ Missing Synthesia credentials.")
        print(f"🔎 Token: {token}")
        print(f"🔎 Avatar ID: {avatar_id}")
        return None

    print(f"✅ Using Synthesia Token: {token[:8]}... (truncated)")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "title": "Strategy Bot Response",
        "description": "MBA chatbot video reply",
        "avatar_id": avatar_id,
        "input": text,
        "voice_id": "en-US-Wavenet-D",
        "visibility": "public",
    }

    print("📤 Sending request to Synthesia API...")
    response = requests.post("https://api.synthesia.io/v2/videos", json=payload, headers=headers)
    print(f"📥 Response status: {response.status_code}")
    print(f"🔎 Response body: {response.text}")

    if response.status_code == 201:
        return response.json().get("video_url")
    else:
        return None

