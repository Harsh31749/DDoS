# ============================================================
# run.py — One-click launcher
# Runs the ML pipeline, then launches the Streamlit dashboard
# Usage: python run.py
# ============================================================

import subprocess
import sys
import time


def main():
    print("=" * 60)
    print("  DDoS DETECTION — AUTO LAUNCHER")
    print("=" * 60)

    # ── Step 1: Run the ML pipeline ───────────────────────────
    print("\n🚀 Step 1: Running ML pipeline (main.py)...")
    print("─" * 60)

    result = subprocess.run(
        [sys.executable, "main.py"],
        check=False
    )

    if result.returncode != 0:
        print("\n❌ Pipeline failed. Fix the errors above before launching the dashboard.")
        sys.exit(1)

    print("\n✅ Pipeline complete! Output files saved to outputs/")

    # ── Step 2: Small pause so files are fully written ────────
    print("\n⏳ Waiting for output files to settle...")
    time.sleep(2)

    # ── Step 3: Launch Streamlit dashboard ────────────────────
    print("\n🌐 Step 2: Launching Streamlit dashboard (app.py)...")
    print("─" * 60)
    print("   Dashboard will open at: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless", "false"],
            check=False
        )
    except KeyboardInterrupt:
        # User pressed Ctrl+C to stop Streamlit — expected, not an error
        print("\n\n👋 Dashboard stopped. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Launcher stopped. Goodbye!")
        sys.exit(0)