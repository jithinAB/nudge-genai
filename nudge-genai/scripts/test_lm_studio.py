#!/usr/bin/env python3
import requests
import json

url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "user", "content": "Say hello in one word"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
}

try:
    print("Testing LM Studio connection...")
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Success! Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.text}")
except requests.exceptions.ConnectionError:
    print("Connection failed - is LM Studio running?")
except requests.exceptions.Timeout:
    print("Request timed out")
except Exception as e:
    print(f"Error: {e}")