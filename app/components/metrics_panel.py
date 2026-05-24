"""
Metric card renderers for the Streamlit dashboard.
"""
import streamlit as st
from typing import Optional


def bpm_metric(bpm: Optional[float], confidence: float, buffer_fill: float):
    """Display heart rate metric with confidence."""
    if bpm is not None:
        delta_str = f"Quality: {confidence*100:.0f}%"
        st.metric(
            label="❤️ Heart Rate",
            value=f"{bpm:.0f} BPM",
            delta=delta_str,
            delta_color="normal",
        )
    else:
        pct = int(buffer_fill * 100)
        st.metric(
            label="❤️ Heart Rate",
            value="Calibrating…",
            delta=f"Buffer: {pct}%",
            delta_color="off",
        )


def fatigue_metric(score: float, state: str, ear: float, perclos: float):
    """Display fatigue score metric."""
    state_emoji = {"Normal": "🟢", "Tired": "🟡", "Drowsy": "🔴"}.get(state, "⚪")
    st.metric(
        label=f"{state_emoji} Fatigue",
        value=f"{score:.0f}%",
        delta=f"EAR: {ear:.3f} | PERCLOS: {perclos:.1f}%",
        delta_color="inverse",
    )


def stress_metric(score: float, state: str):
    """Display stress score metric."""
    state_emoji = {"Calm": "🟢", "Moderate": "🟡", "High": "🔴"}.get(state, "⚪")
    st.metric(
        label=f"{state_emoji} Stress",
        value=f"{score:.0f}%",
        delta=state,
        delta_color="inverse",
    )


def breathing_metric(rate: Optional[float], confidence: float):
    """Display breathing rate metric."""
    if rate is not None and confidence > 0.2:
        st.metric(
            label="🫁 Breathing",
            value=f"{rate:.0f} br/min",
            delta=f"Conf: {confidence*100:.0f}%",
            delta_color="normal",
        )
    else:
        st.metric(
            label="🫁 Breathing",
            value="Estimating…",
            delta="Low signal",
            delta_color="off",
        )


def risk_banner(level: str, level_code: int, messages: list[str]):
    """Display the health risk banner."""
    if level_code == 0:
        st.success(f"**System Status: {level}**  \n" + "  \n".join(messages))
    elif level_code == 1:
        st.warning(f"**System Status: {level}**  \n" + "  \n".join(messages))
    else:
        st.error(f"**System Status: {level}**  \n" + "  \n".join(messages))


def system_status_sidebar(face_detected: bool, cam_fps: float, proc_fps: float, dataset_status: str):
    """Render system status block in the sidebar."""
    st.sidebar.markdown("### System Status")
    face_icon = "🟢" if face_detected else "🔴"
    st.sidebar.markdown(f"{face_icon} **Face Detection**: {'Active' if face_detected else 'No Face'}")
    st.sidebar.markdown(f"📷 **Camera FPS**: {cam_fps:.1f}")
    st.sidebar.markdown(f"⚡ **Process FPS**: {proc_fps:.1f}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Datasets")
    st.sidebar.caption(dataset_status)
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ For educational and research purposes only.  \n"
        "NOT a medical diagnostic device."
    )
