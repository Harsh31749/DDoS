from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_training():
    print("\n🚀 Running ML pipeline (training)...")
    print("─" * 60)

    result = subprocess.run([sys.executable, "main.py"], check=False)

    if result.returncode != 0:
        print("\n❌ Training failed.")
        return False

    print("\n✅ Training complete!")
    return True


def run_realtime():
    print("\n🛡️ Starting real-time detection...")
    print("─" * 60)

    try:
        subprocess.run([sys.executable, "website_monitor.py"], check=False)
    except KeyboardInterrupt:
        print("\n👋 Real-time detection stopped.")


def run_dashboard():
    print("\n🌐 Launching Streamlit dashboard...")
    print("─" * 60)
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop.\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "false",
            ],
            check=False,
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")


def main() -> int:
    print("=" * 60)
    print("  DDoS DETECTION — CONTROL PANEL")
    print("=" * 60)

    print("\nSelect mode:")
    print("1 → Train model")
    print("2 → Real-time detection")
    print("3 → Dashboard only")
    print("4 → Train + Dashboard")
    print("5 → Real-time + Dashboard")

    choice = input("\nEnter choice (1-5): ").strip()

    # --------------------------------------------------
    # MODE 1 — TRAIN ONLY
    # --------------------------------------------------
    if choice == "1":
        run_training()

    # --------------------------------------------------
    # MODE 2 — REALTIME ONLY
    # --------------------------------------------------
    elif choice == "2":
        run_realtime()

    # --------------------------------------------------
    # MODE 3 — DASHBOARD ONLY
    # --------------------------------------------------
    elif choice == "3":
        run_dashboard()

    # --------------------------------------------------
    # MODE 4 — TRAIN + DASHBOARD
    # --------------------------------------------------
    elif choice == "4":
        if run_training():
            time.sleep(1)
            run_dashboard()

    # --------------------------------------------------
    # MODE 5 — REALTIME + DASHBOARD
    # --------------------------------------------------
    elif choice == "5":
        print("\n⚡ Starting real-time detection + dashboard...")

        import threading

        t1 = threading.Thread(target=run_realtime, daemon=True)
        t1.start()

        time.sleep(2)
        run_dashboard()

    else:
        print("❌ Invalid choice.")
        return 1

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped.")
        raise SystemExit(0)