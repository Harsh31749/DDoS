from __future__ import annotations

from pathlib import Path

# ============================================================
# config.py — Central configuration for the DDoS pipeline
# Edit ONLY this file to switch datasets, models, or paths.
# ============================================================

# ── Dataset selection ─────────────────────────────────────────
DATASET_NAME = "CIC-DDoS2019"
LABEL_COLUMN = " Label"  # CIC-DDoS2019 may include a leading-space header
BENIGN_LABEL = "BENIGN"

# ── Mode ─────────────────────────────────────────────────────
DEMO_MODE = False  # False = use real CSVs below

# ── CSV paths ─────────────────────────────────────────────────
CSV_PATHS = [
    "data/DrDoS_DNS.csv",
    "data/DrDoS_LDAP.csv",
    "data/DrDoS_MSSQL.csv",
    "data/DrDoS_NTP.csv",
    "data/DrDoS_NetBIOS.csv",
    "data/DrDoS_SNMP.csv",
    "data/DrDoS_SSDP.csv",
    "data/DrDoS_UDP.csv",
    "data/Syn.csv",
    "data/TFTP.csv",
    "data/UDPLag.csv",
]

SAMPLE_FRAC = 0.10

# ── Attack classes ────────────────────────────────────────────
ATTACK_CLASSES = {
    "CIC-DDoS2019": [
        "DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL",
        "DrDoS_NTP", "DrDoS_NetBIOS", "DrDoS_SNMP",
        "DrDoS_SSDP", "DrDoS_UDP", "Syn", "TFTP",
        "UDPLag", "WebDDoS",
    ],
    "Bot-IoT": [
        "DDoS", "DoS", "Reconnaissance",
        "Data Theft", "Keylogging",
    ],
    "CSE-CIC-IDS2018": [
        "DDoS attack-HOIC", "DDoS attack-LOIC-HTTP",
        "DDoS attacks-LOIC-UDP", "DoS attacks-Hulk",
        "DoS attacks-SlowHTTPTest", "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris", "Bot", "Infiltration",
        "Brute Force-Web", "Brute Force-XSS", "SQL Injection",
    ],
}

# ── Feature selection ─────────────────────────────────────────
TOP_K_FEATURES = 25

# ── Deployment model selection objective ─────────────────────
RECALL_WEIGHT = 0.70
PRECISION_WEIGHT = 0.30

# ── Train / test split ────────────────────────────────────────
TEST_SIZE = 0.30
RANDOM_SEED = 42

# ── Output directory ──────────────────────────────────────────
OUTPUT_DIR = Path("outputs")

# ── Realtime defaults ─────────────────────────────────────────
REALTIME_WINDOW_PACKET_COUNT = 50
REALTIME_ALERT_STREAK_THRESHOLD = 3
REALTIME_ATTACK_CONFIDENCE_THRESHOLD = 0.90
REALTIME_COOLDOWN_SECONDS = 10