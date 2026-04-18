# ============================================================
# config.py — Central configuration for the DDoS pipeline
# Edit ONLY this file to switch datasets, models, or paths.
# ============================================================

import os

# ── Dataset selection ─────────────────────────────────────────
# Options: 'CIC-DDoS2019' | 'Bot-IoT' | 'CSE-CIC-IDS2018'
DATASET_NAME  = "CIC-DDoS2019"
LABEL_COLUMN  = "Label"       # Column name holding class labels
BENIGN_LABEL  = "BENIGN"      # How benign traffic is labelled in your CSV

# ── Mode ─────────────────────────────────────────────────────
# DEMO_MODE = True  → generates 50k synthetic rows (no CSV needed)
# DEMO_MODE = False → reads from CSV_PATHS below
DEMO_MODE = True

# ── CSV paths (used only when DEMO_MODE = False) ──────────────
CSV_PATHS = [
    # "data/DrDoS_DNS.csv",
    # "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# ── Attack classes per dataset ────────────────────────────────
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
TOP_K_FEATURES = 20            # Number of features to keep (InfoGain)

# ── Train / test split ────────────────────────────────────────
TEST_SIZE   = 0.30             # 70% train, 30% test
RANDOM_SEED = 42

# ── Output directory ──────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
