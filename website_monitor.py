import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP
import time
import os
import warnings
import json
import ipaddress
from config import BENIGN_LABEL
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
WINDOW_PACKET_COUNT = 50
ALERT_STREAK_THRESHOLD = 3
BENIGN_LOG_EVERY_WINDOWS = 5
ATTACK_CONFIDENCE_THRESHOLD = 0.90
COOLDOWN_SECONDS = 10

LOG_FILE = "outputs/live_alerts.json"

last_alert_time = {}

print("=" * 60)
print(" 🛡️ REAL-TIME DDoS WEB DEFENSE SHIELD 🛡️")
print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading deployment model and scaler...")

try:
    deployment_config = None

    if os.path.exists("outputs/deployment_config.json"):
        with open("outputs/deployment_config.json", "r") as f:
            deployment_config = json.load(f)

    if deployment_config and deployment_config.get("model_path"):
        model = joblib.load(deployment_config["model_path"])
    else:
        model = joblib.load("outputs/random_forest_model.pkl")

    scaler = joblib.load("outputs/minmax_scaler.pkl")
    selected_features = joblib.load("outputs/selected_features.pkl")

    label_encoder = joblib.load("outputs/label_encoder.pkl") if os.path.exists("outputs/label_encoder.pkl") else None

    print("✅ AI Models loaded successfully.\n")

except FileNotFoundError:
    print("❌ Run 'python run.py' first")
    exit()

active_flows = {}
_warned_missing_encoder = False

# ============================================================
# FEATURE ALIGNMENT (🔥 CRITICAL FIX)
# ============================================================
def align_features(df):
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
    return df[selected_features]

# ============================================================
# HELPERS
# ============================================================
def is_invalid_ip(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return (
            ip_obj.is_private or
            ip_obj.is_loopback or
            ip_obj.is_multicast or
            ip.endswith(".255")
        )
    except:
        return True


def log_attack_to_json(src_ip, prediction):
    alert = {
        "time": time.strftime('%H:%M:%S'),
        "ip": src_ip,
        "attack": prediction,
        "status": "🚨 ATTACK"
    }

    alerts = []

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                alerts = json.load(f)
        except:
            alerts = []

    alerts.insert(0, alert)

    with open(LOG_FILE, "w") as f:
        json.dump(alerts[:50], f, indent=2)


def _decode_prediction(raw_pred):
    global _warned_missing_encoder

    if isinstance(raw_pred, str):
        return raw_pred

    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([int(raw_pred)])[0])
        except:
            pass

    if isinstance(raw_pred, (int, float)):
        if not _warned_missing_encoder:
            print("⚠️ label_encoder missing, fallback decoding used")
            _warned_missing_encoder = True
        return BENIGN_LABEL if int(raw_pred) == 0 else f"ATTACK_{int(raw_pred)}"

    return str(raw_pred)

# ============================================================
# PACKET PROCESSOR
# ============================================================
def process_packet(packet):
    if not (packet.haslayer(IP) and packet.haslayer(TCP)):
        return

    if packet[TCP].dport not in [80, 443] and packet[TCP].sport not in [80, 443]:
        return

    src_ip = packet[IP].src

    if is_invalid_ip(src_ip):
        return

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

    flow = active_flows[src_ip]

    flow["packet_count"] += 1
    flow["total_bytes"] += len(packet)

    if packet[TCP].flags.S:
        flow["syn_count"] += 1
    if packet[TCP].flags.F:
        flow["fin_count"] += 1

    if flow["packet_count"] % WINDOW_PACKET_COUNT != 0:
        return

    duration = max(time.time() - flow["start_time"], 0.001)
    avg_len = flow["total_bytes"] / flow["packet_count"]

    live_data = {col: 0.0 for col in selected_features}

    if "SYN Flag Count" in live_data:
        live_data["SYN Flag Count"] = flow["syn_count"]
    if "Flow Duration" in live_data:
        live_data["Flow Duration"] = duration * 1e6
    if "Flow Packets/s" in live_data:
        live_data["Flow Packets/s"] = flow["packet_count"] / duration
    if "Packet Length Mean" in live_data:
        live_data["Packet Length Mean"] = avg_len

    live_df = pd.DataFrame([live_data])
    live_df = align_features(live_df)

    scaled = scaler.transform(live_df)

    pred = _decode_prediction(model.predict(scaled)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            confidence = float(np.max(model.predict_proba(scaled)[0]))
        except:
            pass

    timestamp = time.strftime('%H:%M:%S')

    if pred != BENIGN_LABEL:

        if confidence and confidence < ATTACK_CONFIDENCE_THRESHOLD:
            flow["attack_streak"] = 0
            return

        flow["attack_streak"] += 1

        if flow["attack_streak"] >= ALERT_STREAK_THRESHOLD:

            now = time.time()

            if src_ip in last_alert_time:
                if now - last_alert_time[src_ip] < COOLDOWN_SECONDS:
                    return

            last_alert_time[src_ip] = now

            print(f"[{timestamp}] 🚨 ATTACK DETECTED from {src_ip} ({pred})")
            log_attack_to_json(src_ip, pred)

    else:
        flow["attack_streak"] = 0
        flow["benign_checks"] += 1

        if flow["benign_checks"] % BENIGN_LOG_EVERY_WINDOWS == 0:
            print(f"[{timestamp}] ✅ {src_ip} is normal")

# ============================================================
# DEMO SERVER (🔥 FIXED)
# ============================================================
class DemoInjectionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass

    def do_POST(self):
        if self.path == '/inject_attack':
            length = int(self.headers['Content-Length'])
            payload = json.loads(self.rfile.read(length))

            df = pd.DataFrame([payload]).fillna(0)

            # 🔥 CRITICAL FIX
            df = align_features(df)

            scaled = scaler.transform(df)

            pred = _decode_prediction(model.predict(scaled)[0])

            print(f"[{time.strftime('%H:%M:%S')}] 🚨 DEMO ATTACK: {pred}")

            log_attack_to_json("DEMO_IP", pred)

            self.send_response(200)
            self.end_headers()

# ============================================================
# START
# ============================================================
def start_sniffer():
    print("🔥 DEBUG MODE: Capturing ALL traffic...")
    sniff(prn=process_packet, store=False)


def start_demo_server():
    server = HTTPServer(('127.0.0.1', 9999), DemoInjectionHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


start_demo_server()
start_sniffer()