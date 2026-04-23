import requests
import time
import random

print("=" * 60)
print(" 💀 DEMO: SIMULATING REAL-TIME DDoS ATTACK")
print("=" * 60)

print("\n🔥 Initiating Attack in 5 seconds. Watch your monitor!")
time.sleep(5)

try:
    for i in range(50):
        # Every 5th request = benign traffic
        if i % 15 == 0:
            payload = {}
        else:
            payload = {
                "packet_count": random.randint(15000, 30000),
                "total_bytes": random.randint(15000000, 40000000),
                "syn_count": random.randint(14000, 29000),
            }

        try:
            requests.post(
                "http://127.0.0.1:9999/inject_attack",
                json=payload,
                timeout=0.1
            )
        except requests.exceptions.RequestException:
            pass

        time.sleep(0.5)

    print("\n🛑 Attack simulation complete.")

except KeyboardInterrupt:
    print("\n🛑 Attack stopped manually.")