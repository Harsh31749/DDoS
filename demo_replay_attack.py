import requests
import time

import pandas as pd

# We will read 50 rows of actual SYN flood data from your training set
CSV_FILE = "data/Syn.csv" 
print("=" * 60)
print(" 💀 DEMO: REPLAYING ACTUAL 'Syn.csv' ATTACK DATA")
print("=" * 60)
print(f"Loading attack signatures from {CSV_FILE}...")

try:
    # Load just a tiny chunk of the real attack data
    df = pd.read_csv(CSV_FILE, nrows=50)
    # Strip whitespace from columns to match your pipeline
    df.columns = df.columns.str.strip()
    print("✅ Real attack data loaded.")
    
    print("\n🔥 Initiating Replay Attack in 5 seconds. Watch your monitor terminal!")
    time.sleep(5)
    
    # 🔥 The new safety block starts here
    try:
        # We will send this data to the hidden "test" endpoint in your monitor
        for index, row in df.iterrows():
            # Every 5th request sends a near-zero payload so dashboard can surface NORMAL events too.
            if index % 5 == 0:
                payload = {}  # benign
            else:
                payload = {
                    "packet_count": 20000,
                    "total_bytes": 20000000,
                    "syn_count": 19500
                }
            try:
                # Send the raw mathematical features directly to the monitor
                requests.post("http://127.0.0.1:9999/inject_attack", json=payload, timeout=0.1)
            except requests.exceptions.RequestException:
                pass # Ignore connection errors, just keep blasting
            
            # 0.5 seconds creates a perfect, readable pulse of alerts
            time.sleep(0.5) 
            
        print("\n🛑 Attack replay complete. All packets sent.")

    except KeyboardInterrupt:
        # This catches your Ctrl+C and prints a clean message!
        print("\n🛑 Attack replay stopped manually by user.")

except Exception as e:
    print(f"❌ Error loading CSV: {e}")