"""
Face Health Digital Twin — Streamlit Dashboard

Entry point: run via  streamlit run app/dashboard.py  (from project root)
or use the wrapper:   python run.py
"""
import sys
import os
import time
from collections import deque
from typing import Optional

# Ensure project root is on path when launched from app/ subdirectory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st
from PIL import Image
import cv2

from core.pipeline import ProcessingPipeline
from app.components.charts import rppg_chart, score_bar_chart, history_chart, vital_gauge
from app.components.metrics_panel import (
    bpm_metric, fatigue_metric, stress_metric,
    breathing_metric, risk_banner, system_status_sidebar,
)
from datasets.loader import DatasetLoader


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Face Health Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the dark professional look
st.markdown("""
<style>
    /* Global */
    .main { background-color: #0E1117; }
    .block-container { padding-top: 1rem; padding-bottom: 0; }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 100%);
        border: 1px solid #00D4FF33;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 16px;
    }
    .dashboard-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00D4FF;
        font-family: monospace;
        letter-spacing: 0.05em;
    }
    .dashboard-subtitle {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-family: monospace;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1A1F2E;
        border: 1px solid #2A3040;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #9CA3AF; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #FAFAFA; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }

    /* Camera feed */
    .stImage img { border-radius: 10px; border: 1px solid #2A3040; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111827; }
    [data-testid="stSidebar"] hr { border-color: #2A3040; }

    /* Alert boxes */
    .stSuccess { background-color: #0A2A1A; border-left: 4px solid #00FF88; }
    .stWarning { background-color: #2A1A00; border-left: 4px solid #FFD700; }
    .stError   { background-color: #2A0A0A; border-left: 4px solid #FF4B4B; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

def _init_state():
    if "pipeline" not in st.session_state:
        pipeline = ProcessingPipeline()
        ok = pipeline.start()
        st.session_state.pipeline = pipeline
        st.session_state.camera_ok = ok

    if "bpm_history" not in st.session_state:
        n = 300  # 5 min at ~1 sample/s
        st.session_state.bpm_history     = deque([0.0] * n, maxlen=n)
        st.session_state.fatigue_history = deque([0.0] * n, maxlen=n)
        st.session_state.stress_history  = deque([0.0] * n, maxlen=n)
        st.session_state.last_bpm        = None
        st.session_state.history_ticker  = 0

    if "dataset_loader" not in st.session_state:
        st.session_state.dataset_loader = DatasetLoader()


_init_state()
pipeline: ProcessingPipeline = st.session_state.pipeline
camera_ok: bool = st.session_state.camera_ok
loader: DatasetLoader = st.session_state.dataset_loader


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 Digital Twin")
    st.markdown("**Face Health Monitor**")
    st.markdown("---")
    refresh_rate = st.slider(
        "Refresh rate (ms)", min_value=100, max_value=1000,
        value=200, step=50,
        help="Lower = smoother but higher CPU usage"
    )
    show_landmarks = st.toggle("Show facial landmarks", value=True)
    st.markdown("---")

    # Sidebar status will be updated in main loop
    sidebar_placeholder = st.empty()


# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------

st.markdown("""
<div class="dashboard-header">
  <div class="dashboard-title">🩺 Face Health Digital Twin</div>
  <div class="dashboard-subtitle">
    Real-time vital sign estimation | rPPG · Fatigue · Stress · Breathing
  </div>
</div>
""", unsafe_allow_html=True)

# Row 1: camera feed + metrics
row1_left, row1_mid, row1_right = st.columns([2, 1, 1])

with row1_left:
    st.markdown("#### 📷 Live Feed")
    camera_placeholder = st.empty()

with row1_mid:
    st.markdown("#### 💓 Vitals")
    hr_placeholder      = st.empty()
    fatigue_placeholder = st.empty()

with row1_right:
    st.markdown("#### 🧠 Mental State")
    stress_placeholder    = st.empty()
    breathing_placeholder = st.empty()

# Row 2: risk banner
risk_placeholder = st.empty()

# Row 3: rPPG waveform + score bars
row3_left, row3_right = st.columns([3, 2])

with row3_left:
    rppg_chart_placeholder = st.empty()

with row3_right:
    score_bar_placeholder = st.empty()

# Row 4: history chart
st.markdown("#### 📈 Vital Trends")
history_chart_placeholder = st.empty()

# Row 5: debug panel (collapsible)
with st.expander("🔍 Advanced Diagnostics", expanded=False):
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    with diag_col1:
        ear_placeholder   = st.empty()
        mar_placeholder   = st.empty()
    with diag_col2:
        blink_placeholder = st.empty()
        hrv_placeholder   = st.empty()
    with diag_col3:
        head_placeholder  = st.empty()
        brow_placeholder  = st.empty()


# ------------------------------------------------------------------
# Helper: convert OpenCV BGR frame → PIL
# ------------------------------------------------------------------

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ------------------------------------------------------------------
# Main update loop
# ------------------------------------------------------------------

def _tick_history(state):
    """Update rolling history at ~1 Hz."""
    st.session_state.history_ticker += 1
    if st.session_state.history_ticker % max(1, int(1000 / refresh_rate)) == 0:
        bpm_val = state.rppg.bpm if state.rppg.bpm else (
            st.session_state.last_bpm or 0.0
        )
        if state.rppg.bpm:
            st.session_state.last_bpm = state.rppg.bpm

        st.session_state.bpm_history.append(bpm_val)
        st.session_state.fatigue_history.append(state.fatigue.score)
        st.session_state.stress_history.append(state.stress.score)


def render(state):
    """Update all dashboard placeholders with the latest pipeline state."""

    # --- Camera feed ---
    if state.annotated_frame is not None:
        if show_landmarks:
            img = _bgr_to_pil(state.annotated_frame)
        else:
            img = _bgr_to_pil(state.frame)
        camera_placeholder.image(img, width="stretch")
    else:
        camera_placeholder.info("📷 Waiting for camera…")

    # --- Vital metrics ---
    with hr_placeholder.container():
        bpm_metric(
            state.rppg.bpm,
            state.rppg.confidence,
            state.rppg.buffer_fill,
        )
    with fatigue_placeholder.container():
        fatigue_metric(
            state.fatigue.score,
            state.fatigue.state,
            state.fatigue.ear,
            state.fatigue.perclos,
        )

    with stress_placeholder.container():
        stress_metric(state.stress.score, state.stress.state)

    with breathing_placeholder.container():
        breathing_metric(state.breathing.rate_bpm, state.breathing.confidence)

    # --- Risk banner ---
    with risk_placeholder.container():
        risk_banner(
            state.risk.level,
            state.risk.level_code,
            state.risk.messages,
        )

    # --- rPPG chart ---
    with rppg_chart_placeholder.container():
        rppg_chart_placeholder.plotly_chart(
            rppg_chart(
                state.rppg.signal,
                state.rppg.timestamps,
                state.rppg.bpm,
                state.rppg.confidence,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    # --- Score bar chart ---
    with score_bar_placeholder.container():
        score_bar_placeholder.plotly_chart(
            score_bar_chart(
                state.fatigue.score,
                state.stress.score,
                state.breathing.rate_bpm,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    # --- History chart ---
    history_chart_placeholder.plotly_chart(
        history_chart(
            list(st.session_state.bpm_history),
            list(st.session_state.fatigue_history),
            list(st.session_state.stress_history),
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    # --- Diagnostics ---
    ear_placeholder.metric("EAR", f"{state.fatigue.ear:.3f}")
    mar_placeholder.metric("MAR", f"{state.fatigue.mar:.3f}")
    blink_placeholder.metric("Blink rate", f"{state.fatigue.blink_rate:.1f} /min")
    hrv_placeholder.metric("HR Variability", f"{state.stress.hr_variability:.1f} BPM")
    head_placeholder.metric("Head movement", f"{state.stress.head_movement:.2f}°/f")
    brow_placeholder.metric("Brow tension", f"{state.stress.brow_tension:.3f}")

    # --- Sidebar status ---
    with sidebar_placeholder.container():
        system_status_sidebar(
            state.face_detected,
            state.camera_fps,
            state.process_fps,
            loader.status_report(),
        )


# --- Camera failure fallback ---
if not camera_ok:
    st.error(
        "**Camera not available.**\n\n"
        "Please check:\n"
        "- A webcam is connected and not used by another app\n"
        "- Camera permissions are granted\n\n"
        "The dashboard will attempt to reconnect. Refresh this page after connecting your webcam."
    )
    time.sleep(2)
    st.rerun()

# --- Main render + auto-refresh ---
state = pipeline.get_state()
_tick_history(state)
render(state)

time.sleep(refresh_rate / 1000.0)
st.rerun()
