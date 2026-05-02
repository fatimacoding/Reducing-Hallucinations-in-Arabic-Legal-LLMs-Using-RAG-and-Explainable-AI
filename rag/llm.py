import requests
import os

# ====== CONFIG ======
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
MODEL = "deepseek-chat"

# ====== GENERATOR ======
def call_deepseek(messages, max_tokens=500):
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]
