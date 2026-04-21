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


def build_live_feature_row(flow: dict[str, Any], selected_features: list[str], duration: float) -> pd.DataFrame:
    live_data = {col: 0.0 for col in selected_features}

    avg_len = flow["total_bytes"] / max(flow["packet_count"], 1)

    # 🔥 EXISTING FEATURES
    if "SYN Flag Count" in live_data:
        live_data["SYN Flag Count"] = float(flow["syn_count"])

    if "Flow Duration" in live_data:
        live_data["Flow Duration"] = float(duration * 1e6)

    if "Flow Packets/s" in live_data:
        live_data["Flow Packets/s"] = float(flow["packet_count"] / max(duration, 0.001))

    if "Packet Length Mean" in live_data:
        live_data["Packet Length Mean"] = float(avg_len)

    # 🔥 NEW FEATURES (CRITICAL FIX)
    packet_rate = flow["packet_count"] / max(duration, 0.001)
    byte_rate = flow["total_bytes"] / max(duration, 0.001)
    syn_ratio = flow["syn_count"] / max(flow["packet_count"], 1)

    if "packet_rate" in live_data:
        live_data["packet_rate"] = float(packet_rate)

    if "byte_rate" in live_data:
        live_data["byte_rate"] = float(byte_rate)

    if "syn_ratio" in live_data:
        live_data["syn_ratio"] = float(syn_ratio)

    return pd.DataFrame([live_data])


def validate_bundle(bundle: dict[str, Any]) -> None:
    required = {"model", "scaler", "features", "label_encoder"}
    missing = sorted(required.difference(bundle.keys()))
    if missing:
        raise ValueError(f"Inference bundle missing required keys: {missing}")
    if not isinstance(bundle["features"], list) or not bundle["features"]:
        raise ValueError("Inference bundle has invalid or empty feature list.")
