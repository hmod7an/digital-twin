"""
Plotly chart builders for the Streamlit dashboard.
All charts use the dark theme to match the dashboard style.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


_DARK_BG   = "#0E1117"
_PANEL_BG  = "#1A1F2E"
_CYAN      = "#00D4FF"
_GREEN     = "#00FF88"
_YELLOW    = "#FFD700"
_RED       = "#FF4B4B"
_ORANGE    = "#FF8C00"
_GRAY      = "#6B7280"


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(color="#FAFAFA", family="monospace"),
        margin=dict(l=40, r=20, t=40, b=30),
        **kwargs,
    )


def rppg_chart(
    signal: np.ndarray,
    timestamps: Optional[np.ndarray],
    bpm: Optional[float],
    confidence: float,
) -> go.Figure:
    """Live rPPG signal waveform with BPM annotation."""
    fig = go.Figure()

    if len(signal) > 5:
        # Show last 10 s of signal
        n = min(len(signal), 300)
        sig = signal[-n:]
        # Normalise for display
        s_min, s_max = sig.min(), sig.max()
        if s_max - s_min > 1e-9:
            sig_norm = (sig - s_min) / (s_max - s_min) * 2 - 1
        else:
            sig_norm = sig - sig.mean()

        t_axis = np.arange(len(sig_norm)) / max(len(sig_norm) - 1, 1)

        fig.add_trace(go.Scatter(
            x=t_axis,
            y=sig_norm,
            mode="lines",
            line=dict(color=_CYAN, width=1.5),
            name="rPPG signal",
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
        ))

    title_text = f"rPPG Signal — {bpm:.0f} BPM" if bpm else "rPPG Signal — calibrating…"
    conf_color = _GREEN if confidence > 0.6 else (_YELLOW if confidence > 0.3 else _RED)

    fig.update_layout(
        **_base_layout(
            title=dict(text=title_text, font=dict(color=_CYAN, size=14)),
            showlegend=False,
            height=180,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                color=_GRAY,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=True,
                gridcolor="#2A3040",
                zeroline=True,
                zerolinecolor=_GRAY,
            ),
        )
    )
    # Confidence badge via annotation
    fig.add_annotation(
        text=f"Quality: {confidence*100:.0f}%",
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        showarrow=False,
        font=dict(color=conf_color, size=11),
        align="right",
    )
    return fig


def vital_gauge(
    value: float,
    title: str,
    unit: str,
    low: float,
    high: float,
    max_val: float,
    color: str = _CYAN,
) -> go.Figure:
    """Gauge indicator for a single vital sign."""
    # Determine needle colour
    if value < low or value > high:
        bar_color = _ORANGE
    else:
        bar_color = color

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=f" {unit}", font=dict(color=bar_color, size=28)),
        title=dict(text=title, font=dict(color="#FAFAFA", size=13)),
        gauge=dict(
            axis=dict(
                range=[0, max_val],
                tickcolor="#6B7280",
                tickfont=dict(color="#6B7280", size=10),
            ),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor=_PANEL_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, low], color="#1E2A3A"),
                dict(range=[low, high], color="#1A3020"),
                dict(range=[high, max_val], color="#3A1A1A"),
            ],
            threshold=dict(
                line=dict(color=_RED, width=2),
                thickness=0.8,
                value=high,
            ),
        ),
    ))
    fig.update_layout(
        **_base_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    )
    return fig


def score_bar_chart(
    fatigue_score: float,
    stress_score: float,
    breathing_rate: Optional[float],
) -> go.Figure:
    """Horizontal bar chart for fatigue, stress, breathing."""
    labels = ["Fatigue", "Stress"]
    values = [fatigue_score, stress_score]
    colors = [
        _score_color(fatigue_score),
        _score_color(stress_score),
    ]

    if breathing_rate is not None:
        # Normalise to 0-100 (normal: 12–20 br/min)
        br_pct = min(100.0, max(0.0, (breathing_rate / 25.0) * 100.0))
        labels.append("Breathing")
        values.append(br_pct)
        colors.append(_CYAN)

    fig = go.Figure()
    for label, val, col in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            x=[val],
            y=[label],
            orientation="h",
            marker_color=col,
            text=[f"{val:.0f}%"],
            textposition="inside",
            textfont=dict(color="white", size=12),
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Vital Scores", font=dict(color="#FAFAFA", size=13)),
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor="#2A3040",
                       tickfont=dict(color=_GRAY)),
            yaxis=dict(showgrid=False, tickfont=dict(color="#FAFAFA", size=12)),
            barmode="overlay",
            height=200,
        )
    )
    return fig


def history_chart(
    bpm_history: list[float],
    fatigue_history: list[float],
    stress_history: list[float],
    x_seconds: int = 30,
) -> go.Figure:
    """Multi-line trend chart for BPM, fatigue, and stress over time."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.05,
    )

    n = len(bpm_history)
    t = list(range(n))

    # BPM line
    if bpm_history:
        fig.add_trace(go.Scatter(
            x=t, y=bpm_history,
            mode="lines",
            line=dict(color=_CYAN, width=2),
            name="BPM",
        ), row=1, col=1)

    # Fatigue + Stress
    if fatigue_history:
        fig.add_trace(go.Scatter(
            x=t, y=fatigue_history,
            mode="lines",
            line=dict(color=_ORANGE, width=2),
            name="Fatigue %",
            fill="tozeroy",
            fillcolor="rgba(255,140,0,0.1)",
        ), row=2, col=1)

    if stress_history:
        fig.add_trace(go.Scatter(
            x=t, y=stress_history,
            mode="lines",
            line=dict(color=_RED, width=2),
            name="Stress %",
        ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=dict(text="Vital History", font=dict(color="#FAFAFA", size=13)),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(color="#FAFAFA", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=280,
        )
    )
    for row in [1, 2]:
        fig.update_xaxes(
            showgrid=True, gridcolor="#2A3040",
            tickfont=dict(color=_GRAY),
            showticklabels=(row == 2),
            row=row, col=1,
        )
    fig.update_yaxes(
        showgrid=True, gridcolor="#2A3040",
        tickfont=dict(color=_GRAY),
        row=1, col=1,
    )
    fig.update_yaxes(
        range=[0, 100],
        showgrid=True, gridcolor="#2A3040",
        tickfont=dict(color=_GRAY),
        row=2, col=1,
    )
    return fig


def _score_color(score: float) -> str:
    if score < 40:
        return _GREEN
    elif score < 65:
        return _YELLOW
    return _RED
