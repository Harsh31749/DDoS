from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT CALL)
# ============================================================
st.set_page_config(
    page_title="DDoS AI Defense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
thead tr th {
    text-align: center !important;
}
tbody tr td {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_CSV = OUTPUT_DIR / "model_performance.csv"
CV_CSV = OUTPUT_DIR / "cv_results.csv"
JSON_FILE = OUTPUT_DIR / "pipeline_results.json"
FEATURE_PNG = OUTPUT_DIR / "feature_importance.png"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrices.png"
METRIC_PNG = OUTPUT_DIR / "metric_comparison.png"
ALERT_FILE = OUTPUT_DIR / "live_alerts.json"
HEALTH_FILE = OUTPUT_DIR / "live_metrics.json"

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data
def load_performance() -> pd.DataFrame | None:
    if not MODEL_CSV.exists():
        st.error(f"File {MODEL_CSV} does not exist.")
        return None
    try:
        return pd.read_csv(MODEL_CSV, index_col=0)
    except Exception as exc:
        st.error(f"Failed to read {MODEL_CSV.name}: {exc}")
        return None

@st.cache_data
def load_cv() -> pd.DataFrame | None:
    if not CV_CSV.exists():
        st.warning(f"File {CV_CSV} does not exist.")
        return None
    try:
        return pd.read_csv(CV_CSV, index_col=0)
    except Exception as exc:
        st.error(f"Failed to read {CV_CSV.name}: {exc}")
        return None

@st.cache_data
def load_payload() -> dict[str, Any] | None:
    if not JSON_FILE.exists():
        st.warning(f"File {JSON_FILE} does not exist.")
        return None
    try:
        with JSON_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Failed to read {JSON_FILE.name}: {exc}")
        return None

def load_alerts() -> list[dict[str, Any]]:
    if not ALERT_FILE.exists():
        return []
    try:
        with ALERT_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data[:25]
    except Exception as exc:
        st.error(f"Failed to read {ALERT_FILE.name}: {exc}")
        return []

def load_health() -> dict[str, Any]:
    if not HEALTH_FILE.exists():
        return {}
    try:
        with HEALTH_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def is_demo_server_running(host: str = "127.0.0.1", port: int = 9999, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_realtime_monitor_running() -> None:
    if ALERT_FILE.exists():
        return

    now = time.time()
    last_try = float(st.session_state.get("monitor_last_start_try", 0.0))
    # Prevent duplicate monitor launches during Streamlit reruns.
    if now - last_try < 10:
        return

    st.session_state["monitor_last_start_try"] = now
    monitor_script = Path(__file__).resolve().parent / "website_monitor.py"
    if not monitor_script.exists():
        st.error("Realtime monitor script `website_monitor.py` not found.")
        return

    try:
        kwargs: dict[str, Any] = {"cwd": str(monitor_script.parent)}
        create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if create_no_window:
            kwargs["creationflags"] = create_no_window
        subprocess.Popen([sys.executable, str(monitor_script)], **kwargs)
        st.info("Starting realtime monitor in background...")
    except Exception as exc:
        st.error(f"Could not start realtime monitor: {exc}")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("🛡️ DDoS Attack Detection Dashboard")
st.caption("Interactive Evaluation of Machine Learning Models for Network Security")
ensure_realtime_monitor_running()

perf_df = load_performance()
cv_df = load_cv()
payload = load_payload()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("⚙️ System Status")
    refresh_sec = st.slider("Refresh every (sec)", 2, 60, value=5)

    st.markdown("---")
    st.subheader("📁 Pipeline Output Files")
    for p in [MODEL_CSV, CV_CSV, JSON_FILE, FEATURE_PNG, CONFUSION_PNG, METRIC_PNG]:
        status = "✅" if p.exists() else "❌"
        st.write(f"{status} `{p.name}`")
    alert_status = "✅" if ALERT_FILE.exists() else "➖"
    st.write(f"{alert_status} `{ALERT_FILE.name}`")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("🧪 Demo Controls")
    if st.button("▶ Run Demo Replay Attack"):
        demo_script = Path(__file__).resolve().parent / "demo_replay_attack.py"
        if not demo_script.exists():
            st.error(f"Script not found: {demo_script.name}")
        elif not is_demo_server_running():
            st.warning("Realtime detector is not running. Start mode 2 or 5 in `run.py`, then retry demo replay.")
        else:
            try:
                # Run in background so Streamlit UI remains responsive.
                kwargs: dict[str, Any] = {"cwd": str(demo_script.parent)}
                create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if create_no_window:
                    kwargs["creationflags"] = create_no_window
                subprocess.Popen([sys.executable, str(demo_script)], **kwargs)
                st.success("Demo replay attack started.")
                st.session_state["show_live_monitoring"] = True
            except Exception as exc:
                st.error(f"Could not start demo replay: {exc}")

st_autorefresh(interval=refresh_sec * 1000, key="dashboard_refresh")

# Stop if no data found
if perf_df is None or perf_df.empty:
    st.error("No model output found. Please run your training pipeline (`python main.py`) first.")
    st.stop()

# Data Cleaning
if "Model" in perf_df.columns:
    perf_df = perf_df.set_index("Model")

metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in perf_df.columns]
for col in metric_cols:
    perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce")

if perf_df.empty:
    st.error("Performance file exists but has no valid rows.")
    st.stop()

# Determine Best Model
if "Recall" in perf_df.columns and perf_df["Recall"].notna().any():
    best_model = perf_df["Recall"].idxmax()
else:
    best_model = perf_df.index[0]

dataset = payload.get("dataset") if payload else "Multi-Attack Dataset"

# ============================================================
# KPI METRICS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Analyzed Dataset", str(dataset))
with c2:
    st.metric("Models Evaluated", len(perf_df.index))
with c3:
    st.metric("🏆 Best Model (Recall)", str(best_model))
with c4:
    best_recall = perf_df.loc[best_model, "Recall"] if "Recall" in perf_df.columns else 0
    try:
        st.metric("Detection Rate", f"{float(best_recall) * 100:.2f}%", delta="Primary Goal")
    except Exception:
        st.metric("Detection Rate", "N/A", delta="Primary Goal")

# ============================================================
# SECTION 1: PERFORMANCE OVERVIEW
# ============================================================
st.markdown("<div class='section-title'>📊 Model Performance Overview</div>", unsafe_allow_html=True)
col_left, col_right = st.columns([1.2, 1])

with col_left:
    highlight_cols = [c for c in ["Recall", "F1-Score"] if c in perf_df.columns]
    styled = perf_df.style
    if highlight_cols:
        styled = styled.highlight_max(axis=0, color="#1e3a8a", subset=highlight_cols)

    st.dataframe(styled, use_container_width=True)
    st.info(f"💡 **Insight:** {best_model} was selected for real-time deployment because it minimizes False Negatives.")

with col_right:
    model_col_name = perf_df.index.name or "Model"
    chart_df = perf_df.reset_index().rename(columns={perf_df.index.name: model_col_name} if perf_df.index.name else {})
    value_vars = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in chart_df.columns]

    if value_vars:
        melted = chart_df.melt(
            id_vars=[model_col_name],
            value_vars=value_vars,
            var_name="Metric",
            value_name="Score",
        )

        fig_bar = px.bar(
            melted,
            x="Metric",
            y="Score",
            color=model_col_name,
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), legend_title_text="Classifier")
        fig_bar.update_traces(textposition="inside", textangle=-90, textfont_size=11)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No metric columns available for charting.")

# ============================================================
# SECTION 2: AI INTELLIGENCE & FEATURES
# ============================================================
st.markdown("<div class='section-title'>🧠 Feature Decision Intelligence</div>", unsafe_allow_html=True)
f_col1, f_col2 = st.columns([1, 1])

with f_col1:
    if payload and isinstance(payload.get("top_features"), dict):
        top_features = (
            pd.DataFrame(list(payload["top_features"].items()), columns=["Feature", "Score"])
            .sort_values("Score", ascending=False)
            .head(15)
        )

        fig_feat = px.bar(
            top_features,
            x="Score",
            y="Feature",
            orientation="h",
            color="Score",
            color_continuous_scale="Tealgrn",
        )
        fig_feat.update_layout(height=500, yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Feature importance metadata not found.")

with f_col2:
    if FEATURE_PNG.exists():
        st.image(str(FEATURE_PNG), caption="Statistical Feature Importance (Saved Image)")
    else:
        st.write(
            "**Impact Summary:** The AI primarily relies on Packet Length Mean and Flow Duration "
            "to distinguish between malicious bot traffic and human users."
        )

# ============================================================
# SECTION 3: ROBUSTNESS & VISUALS
# ============================================================
st.markdown("<div class='section-title'>🧪 Validation & Visual Assets</div>", unsafe_allow_html=True)

v_col1, v_col2 = st.columns(2)
with v_col1:
    if CONFUSION_PNG.exists():
        st.image(str(CONFUSION_PNG), caption="Confusion Matrices (Detection Accuracy per Class)")
    else:
        st.info("Confusion matrix image not found.")

with v_col2:
    if cv_df is not None and not cv_df.empty:
        required = {"CV Recall Mean", "CV Recall Std"}
        if required.issubset(set(cv_df.columns)):
            cv_plot = cv_df.reset_index()
            color_col = "Model" if "Model" in cv_plot.columns else cv_plot.columns[0]
            fig_cv = px.scatter(
                cv_plot,
                x="CV Recall Mean",
                y="CV Recall Std",
                size="CV Recall Mean",
                color=color_col,
                title="Cross-Validation Stability (High Mean + Low Std is Ideal)",
            )
            st.plotly_chart(fig_cv, use_container_width=True)
        else:
            st.info("CV file is present but missing expected columns.")
    else:
        if METRIC_PNG.exists():
            st.image(str(METRIC_PNG), caption="Metric Comparison")
        else:
            st.info("Validation metrics not found.")

st.markdown("---")
st.caption("🛡️ DDoS AI Shield Dashboard")

# ============================================================
# SECTION 4: LIVE ALERT FEED
# ============================================================
alerts = load_alerts()
health = load_health()
realtime_available = ALERT_FILE.exists() or bool(health)
show_live_monitoring = True

if show_live_monitoring:
    st.markdown("### 🟢 System Status")
    if realtime_available:
        st.success("🛡️ Monitoring Active (Realtime Data Available)")
        st.markdown("<div class='section-title'>🚨 Real-Time Security Events</div>", unsafe_allow_html=True)
    else:
        st.info("Realtime detector is not running yet. Start mode 2 or 5 from `run.py`.")

    if realtime_available and alerts:
        alert_df = pd.DataFrame(alerts)
        if "ip" in alert_df.columns:
            ip_options = ["All"] + sorted(alert_df["ip"].dropna().astype(str).unique().tolist())
            selected_ip = st.selectbox("Filter by source IP", ip_options, index=0)
            if selected_ip != "All":
                alert_df = alert_df[alert_df["ip"].astype(str) == selected_ip]

        if "attack" in alert_df.columns:
            attack_options = ["All"] + sorted(alert_df["attack"].dropna().astype(str).unique().tolist())
            selected_attack = st.selectbox("Filter by attack type", attack_options, index=0)
            if selected_attack != "All":
                alert_df = alert_df[alert_df["attack"].astype(str) == selected_attack]

        def color_status(val: Any) -> str:
            text = str(val).upper()
            color = "#ef4444" if text == "ATTACK" else "#10b981"
            return f"color: {color}; font-weight: bold"

        if "status" in alert_df.columns:
            styled = alert_df.style
            try:
                styled = styled.map(color_status, subset=["status"])
            except AttributeError:
                styled = styled.applymap(color_status, subset=["status"])
            styled = styled.set_properties(**{"text-align": "center"})
            styled = styled.set_table_styles(
                [
                    {"selector": "th", "props": [("text-align", "center")]},
                    {"selector": "td", "props": [("text-align", "center")]},
                ]
            )
            st.dataframe(styled, use_container_width=True)
        if alerts:
            latest = alerts[0]
            latest_status = str(latest.get("status", "ATTACK")).upper()
            if latest_status == "ATTACK":
                st.error(f"🚨 Latest Event → IP: {latest.get('ip', 'N/A')} | Time: {latest.get('time', 'N/A')}")
            else:
                st.success(f"✅ Latest Event → IP: {latest.get('ip', 'N/A')} | Time: {latest.get('time', 'N/A')}")
        else:
            st.table(alert_df)

        if st.button("🗑️ Clear Alert History"):
            try:
                if ALERT_FILE.exists():
                    ALERT_FILE.unlink()
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not clear alert file: {exc}")

        if "ip" in alert_df.columns and not alert_df.empty:
            st.markdown("#### Top Source IPs")
            ip_counts = alert_df["ip"].astype(str).value_counts().head(10)
            top_ips = pd.DataFrame({"IP": ip_counts.index, "Alerts": ip_counts.values})
            fig_ips = px.bar(top_ips, x="IP", y="Alerts", color="Alerts", color_continuous_scale="Reds")
            st.plotly_chart(fig_ips, use_container_width=True)
    elif realtime_available:
        if ALERT_FILE.exists():
            st.success("✅ System Secure: No active threats detected in the last window.")
        else:
            st.success("✅ No active alerts in the current window.")
if show_live_monitoring and health:
    st.markdown("### 🩺 Runtime Health")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Predictions / min", health.get("predictions_per_min", "N/A"))
    h2.metric("Alerts / min", health.get("alerts_per_min", "N/A"))
    h3.metric("Tracked Flows", health.get("flows_tracked", "N/A"))
    h4.metric("Error Count", health.get("error_count", "N/A"))