from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scapy.all import IP, TCP, sniff

from config import (
    BENIGN_LABEL,
    REALTIME_ALERT_STREAK_THRESHOLD,
    REALTIME_ATTACK_CONFIDENCE_THRESHOLD,
    REALTIME_COOLDOWN_SECONDS,
    REALTIME_WINDOW_PACKET_COUNT,
)
from inference_contract import align_features, build_live_feature_row, validate_bundle

WINDOW_PACKET_COUNT = REALTIME_WINDOW_PACKET_COUNT
ALERT_STREAK_THRESHOLD = REALTIME_ALERT_STREAK_THRESHOLD
BENIGN_LOG_EVERY_WINDOWS = 5
ATTACK_CONFIDENCE_THRESHOLD = REALTIME_ATTACK_CONFIDENCE_THRESHOLD
COOLDOWN_SECONDS = REALTIME_COOLDOWN_SECONDS
FLOW_TTL_SECONDS = 120
MAX_TRACKED_FLOWS = 5000
SNIFF_FILTER = "tcp and (port 80 or port 443)"
ENABLE_DEMO_SERVER = True
DEMO_TOKEN = os.getenv("DDoS_DEMO_TOKEN", "")

LOG_FILE = "outputs/live_alerts.json"
HEALTH_FILE = "outputs/live_metrics.json"

active_flows: dict[str, dict[str, Any]] = {}
last_alert_time: dict[str, float] = {}
runtime = defaultdict(float)
runtime["start_time"] = time.time()
runtime["windows_processed"] = 0
runtime["alerts_emitted"] = 0
runtime["predictions"] = 0
runtime["errors"] = 0

print("=" * 60)
print(" REAL-TIME DDoS WEB DEFENSE SHIELD ")
print("=" * 60)

print("Loading inference bundle...")
bundle = joblib.load("outputs/inference_bundle.pkl")
validate_bundle(bundle)
model = bundle["model"]
scaler = bundle["scaler"]
imputer = bundle.get("imputer")
selected_features = bundle["features"]
label_encoder = bundle.get("label_encoder")
model_name = bundle.get("model_name", "unknown")
print(f"Bundle loaded successfully. Model: {model_name}")


def is_invalid_ip(ip: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_multicast or ip.endswith(".255")
    except ValueError:
        return True


def decode_prediction(raw_pred: Any) -> str:
    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([int(raw_pred)])[0])
        except Exception:
            pass
    try:
        return BENIGN_LABEL if int(raw_pred) == 0 else f"ATTACK_{int(raw_pred)}"
    except Exception:
        return str(raw_pred)


def log_attack_to_json(src_ip: str, prediction: str, confidence: float | None) -> None:
    alert = {
        "time": time.strftime("%H:%M:%S"),
        "ip": src_ip,
        "attack": prediction,
        "confidence": None if confidence is None else round(confidence, 4),
        "status": "ATTACK",
    }
    alerts = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                alerts = json.load(f)
            if not isinstance(alerts, list):
                alerts = []
        except Exception:
            alerts = []

    alerts.insert(0, alert)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts[:250], f, indent=2)


def update_health() -> None:
    elapsed = max(time.time() - runtime["start_time"], 1.0)
    payload = {
        "uptime_s": round(elapsed, 2),
        "flows_tracked": len(active_flows),
        "windows_processed": int(runtime["windows_processed"]),
        "predictions": int(runtime["predictions"]),
        "alerts_emitted": int(runtime["alerts_emitted"]),
        "error_count": int(runtime["errors"]),
        "predictions_per_min": round(runtime["predictions"] * 60.0 / elapsed, 3),
        "alerts_per_min": round(runtime["alerts_emitted"] * 60.0 / elapsed, 3),
        "last_updated": time.strftime("%H:%M:%S"),
        "model_name": model_name,
    }
    os.makedirs("outputs", exist_ok=True)
    with open(HEALTH_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def prune_flows() -> None:
    now = time.time()
    to_remove = [ip for ip, flow in active_flows.items() if now - flow["last_seen"] > FLOW_TTL_SECONDS]
    for ip in to_remove:
        active_flows.pop(ip, None)

    if len(active_flows) <= MAX_TRACKED_FLOWS:
        return
    sorted_items = sorted(active_flows.items(), key=lambda item: item[1]["last_seen"])
    for ip, _ in sorted_items[: len(active_flows) - MAX_TRACKED_FLOWS]:
        active_flows.pop(ip, None)


def should_alert(prediction_label: str, confidence: float | None) -> bool:
    is_attack_label = prediction_label != BENIGN_LABEL
    if not is_attack_label:
        return False
    if confidence is None:
        return True
    return confidence >= ATTACK_CONFIDENCE_THRESHOLD


def process_packet(packet) -> None:
    try:
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return
        src_ip = packet[IP].src
        if is_invalid_ip(src_ip):
            return

        flow = active_flows.setdefault(
            src_ip,
            {
                "start_time": time.time(),
                "last_seen": time.time(),
                "packet_count": 0,
                "syn_count": 0,
                "fin_count": 0,
                "total_bytes": 0,
                "attack_streak": 0,
                "benign_checks": 0,
            },
        )

        flow["last_seen"] = time.time()
        flow["packet_count"] += 1
        flow["total_bytes"] += len(packet)

        flags = int(packet[TCP].flags)
        if flags & 0x02:
            flow["syn_count"] += 1
        if flags & 0x01:
            flow["fin_count"] += 1

        if flow["packet_count"] % WINDOW_PACKET_COUNT != 0:
            return

        duration = max(time.time() - flow["start_time"], 0.001)
        live_df = build_live_feature_row(flow, selected_features, duration)
        live_df = align_features(live_df, selected_features)
        if imputer is not None:
            live_df = pd.DataFrame(imputer.transform(live_df), columns=live_df.columns)
        scaled = scaler.transform(live_df)

        raw_pred = model.predict(scaled)[0]
        pred = decode_prediction(raw_pred)

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            confidence = float(np.max(proba))
        runtime["windows_processed"] += 1
        runtime["predictions"] += 1

        timestamp = time.strftime("%H:%M:%S")
        if should_alert(pred, confidence):
            flow["attack_streak"] += 1
            if flow["attack_streak"] >= ALERT_STREAK_THRESHOLD:
                now = time.time()
                if now - last_alert_time.get(src_ip, 0.0) >= COOLDOWN_SECONDS:
                    last_alert_time[src_ip] = now
                    runtime["alerts_emitted"] += 1
                    print(f"[{timestamp}] ATTACK from {src_ip} ({pred}, conf={confidence})")
                    log_attack_to_json(src_ip, pred, confidence)
        else:
            flow["attack_streak"] = 0
            flow["benign_checks"] += 1
            if flow["benign_checks"] % BENIGN_LOG_EVERY_WINDOWS == 0:
                print(f"[{timestamp}] benign traffic from {src_ip}")

        prune_flows()
        update_health()
    except Exception as exc:
        runtime["errors"] += 1
        print(f"Processing error: {exc}")


class DemoInjectionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path != "/inject_attack":
            self.send_response(404)
            self.end_headers()
            return

        if DEMO_TOKEN:
            incoming = self.headers.get("X-Demo-Token", "")
            if incoming != DEMO_TOKEN:
                self.send_response(401)
                self.end_headers()
                return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 100_000:
            self.send_response(400)
            self.end_headers()
            return

        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            numeric_payload = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
            df = pd.DataFrame([numeric_payload]).fillna(0)
            df = align_features(df, selected_features)
            if imputer is not None:
                df = pd.DataFrame(imputer.transform(df), columns=df.columns)
            scaled = scaler.transform(df)
            pred = decode_prediction(model.predict(scaled)[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(scaled)[0]))
            log_attack_to_json("DEMO_IP", pred, confidence)
            self.send_response(200)
            self.end_headers()
        except Exception:
            self.send_response(400)
            self.end_headers()


def start_demo_server() -> None:
    if not ENABLE_DEMO_SERVER:
        return
    server = HTTPServer(("127.0.0.1", 9999), DemoInjectionHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Demo injection server running on http://127.0.0.1:9999/inject_attack")


def start_sniffer() -> None:
    print(f"Sniffer started with filter: {SNIFF_FILTER}")
    sniff(filter=SNIFF_FILTER, prn=process_packet, store=False)


start_demo_server()
start_sniffer()
