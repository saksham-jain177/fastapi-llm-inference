import requests
import json
import time

URL = "http://localhost:8000/feedback"

payload = {
    "query": "Define RLHF",
    "response": "RLHF stands for Reinforcement Learning from Human Feedback.",
    "rating": 1,
    "model_mode": "quantized"
}

print(f"Sending feedback to {URL}...")
try:
    response = requests.post(URL, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200 and response.json().get("status") == "recorded":
        print("SUCCESS: Feedback recorded.")
    else:
        print("FAILURE: API did not return success status.")
        
except Exception as e:
    print(f"ERROR: Could not connect to API. {e}")
