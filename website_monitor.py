import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, conf
import time
import os
import warnings
import json
from config import BENIGN_LABEL

# Suppress warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)
WINDOW_PACKET_COUNT = 50
ALERT_STREAK_THRESHOLD = 3
BENIGN_LOG_EVERY_WINDOWS = 5

print("=" * 60)
print(" 🛡️ REAL-TIME DDoS WEB DEFENSE SHIELD 🛡️")
print("=" * 60)

# 1. Load the AI Brain
print("Loading deployment model and scaler...")
try:
    deployment_config = None
    if os.path.exists("outputs/deployment_config.json"):
        with open("outputs/deployment_config.json", "r", encoding="utf-8") as f:
            deployment_config = json.load(f)

    if deployment_config and deployment_config.get("model_path"):
        selected_model_name = deployment_config.get("selected_model", "Unknown")
        model_path = deployment_config["model_path"]
        selection_metric = deployment_config.get("selection_metric", "unknown")
        print(
            f"Loading deployment-selected model: {selected_model_name} ({model_path}) "
            f"[policy={selection_metric}]"
        )
        model = joblib.load(model_path)
    else:
        # Backward compatibility with older outputs
        print("No deployment_config.json found. Falling back to outputs/random_forest_model.pkl")
        model = joblib.load("outputs/random_forest_model.pkl")

    scaler = joblib.load("outputs/minmax_scaler.pkl")
    features = joblib.load("outputs/selected_features.pkl")
    label_encoder = joblib.load("outputs/label_encoder.pkl") if os.path.exists("outputs/label_encoder.pkl") else None
    print("✅ AI Models loaded successfully.\n")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run 'python main.py' first.")
    exit()

# Tracker for active connections
active_flows = {}
_warned_missing_encoder = False


def _decode_prediction(raw_pred):
    """
    Convert model output into a human-readable class label.
    Handles integer-encoded classes from training.
    """
    global _warned_missing_encoder

    # If model already returns label string, use it directly.
    if isinstance(raw_pred, str):
        return raw_pred

    # Decode integer class index using saved LabelEncoder when available.
    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([int(raw_pred)])[0])
        except Exception:
            pass

    # Backward compatibility: old artifacts may not include label_encoder.pkl.
    # Most binary models use 0 as benign and 1 as attack.
    if isinstance(raw_pred, (int, np.integer, float, np.floating)):
        idx = int(raw_pred)
        if not _warned_missing_encoder:
            print("⚠️ label_encoder.pkl not found. Using fallback class decoding (0 -> BENIGN).")
            _warned_missing_encoder = True
        return BENIGN_LABEL if idx == 0 else f"ATTACK_CLASS_{idx}"

    return str(raw_pred)

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
            "attack_streak": 0,
            "benign_checks": 0,
        }

    # Update flow statistics
    flow = active_flows[src_ip]
    flow["packet_count"] += 1
    flow["total_bytes"] += len(packet)
    
    if packet[TCP].flags.S: # SYN Flag
        flow["syn_count"] += 1
    if packet[TCP].flags.F: # FIN Flag
        flow["fin_count"] += 1
        
    # Every N packets from an IP, run a security check.
    if flow["packet_count"] % WINDOW_PACKET_COUNT == 0:
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
        raw_prediction = model.predict(scaled_data)[0]
        prediction = _decode_prediction(raw_prediction)
        
        if prediction != BENIGN_LABEL:
            flow["attack_streak"] += 1
            if flow["attack_streak"] >= ALERT_STREAK_THRESHOLD:
                print(
                    f"🚨 [ATTACK DETECTED] {prediction} flood from {src_ip}! "
                    f"(streak={flow['attack_streak']}) Triggering firewall..."
                )
                # Here is where the firewall block command would go
            else:
                print(
                    f"⚠️ Suspicious traffic from {src_ip}: predicted {prediction} "
                    f"(streak={flow['attack_streak']}/{ALERT_STREAK_THRESHOLD})"
                )
        else:
            recovered = flow["attack_streak"] > 0
            flow["attack_streak"] = 0
            flow["benign_checks"] += 1
            if recovered or flow["benign_checks"] % BENIGN_LOG_EVERY_WINDOWS == 0:
                print(
                    f"✅ Traffic from {src_ip} is benign. "
                    f"(Packets: {flow['packet_count']}, checks={flow['benign_checks']})"
                )

print("Listening for incoming website traffic on Ports 80 and 443. Press Ctrl+C to exit.")


def _is_web_tcp(packet) -> bool:
    """Python-level packet filter used when BPF/libpcap is unavailable."""
    return (
        packet.haslayer(IP)
        and packet.haslayer(TCP)
        and (packet[TCP].dport in [80, 443] or packet[TCP].sport in [80, 443])
    )


def start_sniffer() -> None:
    """
    Start packet sniffing with a graceful fallback for Windows systems
    missing WinPcap/Npcap drivers.
    """
    try:
        # Fast path: BPF kernel filter (requires pcap provider)
        sniff(filter="tcp port 80 or tcp port 443", prn=process_packet, store=False)
    except RuntimeError as err:
        err_text = str(err)
        if "winpcap is not installed" in err_text.lower():
            print("\n⚠️ Packet capture driver missing on Windows (WinPcap/Npcap).")
            print("   Attempting Layer-3 fallback without BPF filter...")
            print("   For best results, install Npcap and run terminal as Administrator.\n")
            try:
                # Fallback path: no BPF filter + Python-level filtering.
                # Works in more environments but may capture extra traffic and be slower.
                sniff(
                    prn=process_packet,
                    lfilter=_is_web_tcp,
                    L3socket=conf.L3socket,
                    store=False,
                )
            except Exception as fallback_err:
                print("❌ Fallback sniffing failed.")
                print(f"   Reason: {fallback_err}")
                print("   Fix:")
                print("   1) Install Npcap (https://npcap.com) with WinPcap compatibility enabled.")
                print("   2) Re-run this script from an Administrator shell.")
        else:
            raise


start_sniffer()