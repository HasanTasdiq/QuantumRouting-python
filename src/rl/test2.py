import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"  # Change to your API server address
INTERVAL = 10  # seconds between memory checks
TAKE_SNAPSHOT = True  # Set True to also get top memory allocations

def get_memory():
    try:
        r = requests.get(f"{BASE_URL}/debug_memory")
        data = r.json()
        print(f"[Memory] RSS: {data['rss_MB']} MB")
    except Exception as e:
        print("Error fetching memory:", e)

def get_snapshot():
    try:
        r = requests.get(f"{BASE_URL}/snapshot")
        data = r.json()
        print("[Snapshot] Top memory allocations:")
        for line in data["top"]:
            print(" ", line)
    except Exception as e:
        print("Error fetching snapshot:", e)

if __name__ == "__main__":
    while True:
        print(f"\n=== Checking memory at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        get_memory()
        if TAKE_SNAPSHOT:
            get_snapshot()
        time.sleep(INTERVAL)
