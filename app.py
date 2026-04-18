from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="DDoS Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = Path("outputs")
MODEL_CSV = OUTPUT_DIR / "model_performance.csv"
CV_CSV = OUTPUT_DIR / "cv_results.csv"
JSON_FILE = OUTPUT_DIR / "pipeline_results.json"
FEATURE_PNG = OUTPUT_DIR / "feature_importance.png"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrices.png"
METRIC_PNG = OUTPUT_DIR / "metric_comparison.png"


@st.cache_data
def load_performance() -> pd.DataFrame | None:
    if not MODEL_CSV.exists():
        return None
    return pd.read_csv(MODEL_CSV, index_col=0)


@st.cache_data
def load_cv() -> pd.DataFrame | None:
    if not CV_CSV.exists():
        return None
    return pd.read_csv(CV_CSV, index_col=0)


@st.cache_data
def load_payload() -> dict[str, Any] | None:
    if not JSON_FILE.exists():
        return None
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)



st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem;}
        .metric-card {
            background: linear-gradient(120deg, #0f172a, #1e293b);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 18px;
            color: #e2e8f0;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ DDoS Attack Detection Dashboard")
st.caption("Interactive monitoring and evaluation dashboard powered by your training outputs.")

perf_df = load_performance()
cv_df = load_cv()
payload = load_payload()

with st.sidebar:
    st.header("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (manual rerun friendly)", value=False)
    if auto_refresh:
        st.info("Use Streamlit's rerun button to refresh live files.")

    st.markdown("---")
    st.subheader("📁 Output files")
    for p in [MODEL_CSV, CV_CSV, JSON_FILE, FEATURE_PNG, CONFUSION_PNG, METRIC_PNG]:
        st.write(f"{'✅' if p.exists() else '❌'} `{p}`")

if perf_df is None:
    st.warning("No model output found yet. Run `python main.py` first to generate files in `outputs/`.")
    st.stop()

if "Model" in perf_df.columns:
    perf_df = perf_df.set_index("Model")

for col in ["Accuracy", "Precision", "Recall", "F1-Score", "Train (s)", "Infer (s)"]:
    if col in perf_df.columns:
        perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce")

best_model = perf_df["Recall"].astype(float).idxmax() if "Recall" in perf_df.columns else perf_df.index[0]
dataset = payload.get("dataset") if payload else "Unknown"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Dataset", dataset)
with c2:
    st.metric("Models Trained", len(perf_df.index))
with c3:
    st.metric("Best Recall Model", best_model)
with c4:
    best_recall = perf_df.loc[best_model, "Recall"] if "Recall" in perf_df.columns else None
    st.metric("Best Recall", f"{best_recall:.4f}" if pd.notnull(best_recall) else "N/A")

st.markdown("<div class='section-title'>📊 Model Performance Overview</div>", unsafe_allow_html=True)
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.dataframe(perf_df, use_container_width=True)

with col_right:
    melted = perf_df.reset_index().melt(
        id_vars=[perf_df.index.name or "index"],
        value_vars=[c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in perf_df.columns],
        var_name="Metric",
        value_name="Score",
    )
    model_col = perf_df.index.name or "index"
    fig_bar = px.bar(
        melted,
        x="Metric",
        y="Score",
        color=model_col,
        barmode="group",
        text_auto=".3f",
        title="Metric Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(height=420, legend_title_text="Model")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<div class='section-title'>🧠 Feature Intelligence</div>", unsafe_allow_html=True)
f_col1, f_col2 = st.columns([1, 1])

with f_col1:
    if payload and "top_features" in payload:
        top_features = pd.DataFrame(
            list(payload["top_features"].items()), columns=["Feature", "Score"]
        ).sort_values("Score", ascending=False)
        fig_feat = px.bar(
            top_features,
            x="Score",
            y="Feature",
            orientation="h",
            title="Top 20 Mutual Information Features",
            color="Score",
            color_continuous_scale="Viridis",
        )
        fig_feat.update_layout(height=560, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("`pipeline_results.json` missing top feature metadata.")

with f_col2:
    if FEATURE_PNG.exists():
        st.image(str(FEATURE_PNG), caption="Feature importance image exported by training pipeline")
    else:
        st.info("Feature importance image not available yet.")

st.markdown("<div class='section-title'>🧪 Cross-Validation Stability</div>", unsafe_allow_html=True)
if cv_df is not None and not cv_df.empty:
    if "Model" in cv_df.columns:
        cv_df = cv_df.set_index("Model")

    for c in cv_df.columns:
        cv_df[c] = pd.to_numeric(cv_df[c], errors="coerce")

    cv_plot = cv_df.reset_index()
    model_col = cv_df.index.name or "index"
    fig_cv = px.scatter(
        cv_plot,
        x="CV F1 Mean",
        y="CV F1 Std",
        size="CV F1 Mean",
        color=model_col,
        hover_name=model_col,
        title="CV Robustness (high mean, low std preferred)",
        size_max=32,
    )
    fig_cv.update_layout(height=420)

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(fig_cv, use_container_width=True)
    with right:
        st.dataframe(cv_df, use_container_width=True)
else:
    st.info("No cross-validation file found. Generate it via `python main.py`.")

st.markdown("<div class='section-title'>🖼️ Exported Visual Assets</div>", unsafe_allow_html=True)
v1, v2 = st.columns(2)
with v1:
    if CONFUSION_PNG.exists():
        st.image(str(CONFUSION_PNG), caption="Confusion matrices")
    else:
        st.info("`outputs/confusion_matrices.png` not found.")
with v2:
    if METRIC_PNG.exists():
        st.image(str(METRIC_PNG), caption="Metric comparison")
    else:
        st.info("`outputs/metric_comparison.png` not found.")

st.markdown("---")
st.caption("Tip: Run with `streamlit run app.py` after training completes.")