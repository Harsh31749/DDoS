from __future__ import annotations

from typing import Any

import pandas as pd

BUNDLE_SCHEMA_VERSION = "1.1.0"


def align_features(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    working = df.copy()
    for col in selected_features:
        if col not in working.columns:
            working[col] = 0.0
    return working[selected_features]


def build_live_feature_row(flow, selected_features, duration):
    live_data = {col: 0.0 for col in selected_features}

    pkt = flow["packet_count"]
    bytes_ = flow["total_bytes"]
    syn = flow["syn_count"]

    duration = max(duration, 0.001)

    avg_len = bytes_ / max(pkt, 1)

    packet_rate = pkt / duration
    byte_rate = bytes_ / duration
    syn_ratio = syn / max(pkt, 1)

    # 🔥 CORE FEATURES
    if "Flow Duration" in live_data:
        live_data["Flow Duration"] = duration * 1e6

    if "Flow Packets/s" in live_data:
        live_data["Flow Packets/s"] = packet_rate

    if "Flow Bytes/s" in live_data:
        live_data["Flow Bytes/s"] = byte_rate

    if "packet_rate" in live_data:
        live_data["packet_rate"] = packet_rate

    if "byte_rate" in live_data:
        live_data["byte_rate"] = byte_rate

    if "syn_ratio" in live_data:
        live_data["syn_ratio"] = syn_ratio

    if "SYN Flag Count" in live_data:
        live_data["SYN Flag Count"] = syn

    # 🔥 CRITICAL PACKET LENGTH FEATURES
    if "Packet Length Mean" in live_data:
        live_data["Packet Length Mean"] = avg_len

    if "Min Packet Length" in live_data:
        live_data["Min Packet Length"] = avg_len * 0.5

    if "Max Packet Length" in live_data:
        live_data["Max Packet Length"] = avg_len * 1.5

    if "Fwd Packet Length Max" in live_data:
        live_data["Fwd Packet Length Max"] = avg_len * 1.5

    if "Fwd Packet Length Min" in live_data:
        live_data["Fwd Packet Length Min"] = avg_len * 0.5

    if "Total Length of Fwd Packets" in live_data:
        live_data["Total Length of Fwd Packets"] = bytes_

    return pd.DataFrame([live_data])


def validate_bundle(bundle: dict[str, Any]) -> None:
    required = {"model", "scaler", "features", "label_encoder"}
    missing = sorted(required.difference(bundle.keys()))
    if missing:
        raise ValueError(f"Inference bundle missing required keys: {missing}")
    if not isinstance(bundle["features"], list) or not bundle["features"]:
        raise ValueError("Inference bundle has invalid or empty feature list.")
