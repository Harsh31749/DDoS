import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP
import time
import os
import warnings

# Suppress warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)

print("=" * 60)
print(" 🛡️ REAL-TIME DDoS WEB DEFENSE SHIELD 🛡️")
print("=" * 60)

# 1. Load the AI Brain
print("Loading Random Forest model and scaler...")
try:
    model = joblib.load("outputs/random_forest_model.pkl")
    scaler = joblib.load("outputs/minmax_scaler.pkl")
    features = joblib.load("outputs/selected_features.pkl")
    print("✅ AI Models loaded successfully.\n")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run 'python main.py' first.")
    exit()

# Tracker for active connections
active_flows = {}

def process_packet(packet):
    # We only care about TCP web traffic (HTTP Port 80, HTTPS Port 443)
    if not (packet.haslayer(IP) and packet.haslayer(TCP)):
        return
    if packet[TCP].dport not in [80, 443] and packet[TCP].sport not in [80, 443]:
        return

    src_ip = packet[IP].src
    
    # Initialize a new flow if we haven't seen this IP recently
    if src_ip not in active_flows:
        active_flows[src_ip] = {
            "start_time": time.time(),
            "packet_count": 0,
            "syn_count": 0,
            "fin_count": 0,
            "total_bytes": 0,
        }

    # Update flow statistics
    flow = active_flows[src_ip]
    flow["packet_count"] += 1
    flow["total_bytes"] += len(packet)
    
    if packet[TCP].flags.S: # SYN Flag
        flow["syn_count"] += 1
    if packet[TCP].flags.F: # FIN Flag
        flow["fin_count"] += 1
        
    # Every 50 packets from an IP, run a security check!
    if flow["packet_count"] % 50 == 0:
        duration = time.time() - flow["start_time"]
        if duration == 0: duration = 0.001 # Prevent division by zero
        
        # Build an empty row with 0s for all 20 features
        live_data = {col: 0.0 for col in features}
        
        # Populate the features we can easily calculate in real-time
        if "SYN Flag Count" in live_data: live_data["SYN Flag Count"] = flow["syn_count"]
        if "FIN Flag Count" in live_data: live_data["FIN Flag Count"] = flow["fin_count"]
        if "Flow Duration" in live_data: live_data["Flow Duration"] = duration * 1e6 # Microseconds
        if "Flow Packets/s" in live_data: live_data["Flow Packets/s"] = flow["packet_count"] / duration
        if "Flow Bytes/s" in live_data: live_data["Flow Bytes/s"] = flow["total_bytes"] / duration
        if "Total Fwd Packets" in live_data: live_data["Total Fwd Packets"] = flow["packet_count"]
        
        # Prepare for AI
        live_df = pd.DataFrame([live_data], columns=features)
        scaled_data = pd.DataFrame(scaler.transform(live_df), columns=features)
        
        # The AI makes its decision
        prediction = model.predict(scaled_data)[0]
        
        if prediction != "BENIGN":
            print(f"🚨 [ATTACK DETECTED] {prediction} flood from {src_ip}! Triggering firewall...")
            # Here is where the firewall block command would go
        else:
            print(f"✅ Traffic from {src_ip} is benign. (Packets: {flow['packet_count']})")

print("Listening for incoming website traffic on Ports 80 and 443. Press Ctrl+C to exit.")
# Sniff traffic continuously
sniff(filter="tcp port 80 or tcp port 443", prn=process_packet, store=False)