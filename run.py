from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("  DDoS DETECTION — AUTO LAUNCHER")
    print("=" * 60)

    if not Path("main.py").exists():
        print("❌ main.py not found in current directory.")
        return 1
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory.")
        return 1

    # Step 1: Run ML pipeline
    print("\n🚀 Step 1: Running ML pipeline (main.py)...")
    print("─" * 60)

    result = subprocess.run([sys.executable, "main.py"], check=False)
    if result.returncode != 0:
        print("\n❌ Pipeline failed. Fix errors above before launching dashboard.")
        return result.returncode

    print("\n✅ Pipeline complete! Output files saved to outputs/")

    # Step 2: brief delay
    print("\n⏳ Waiting for output files to settle...")
    time.sleep(1)

    # Step 3: Launch Streamlit
    print("\n🌐 Step 2: Launching Streamlit dashboard (app.py)...")
    print("─" * 60)
    print("   Dashboard URL: http://localhost:8501")
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
        return 0

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped.")
        raise SystemExit(0)