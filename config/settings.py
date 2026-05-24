"""
Central configuration for the Face Health Digital Twin system.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EmotionConfig:
    ema_alpha: float = 0.12            # EMA smoothing (lower = more stable)
    min_frames_to_change: int = 20     # frames candidate must dominate before commit
    confidence_threshold: float = 0.22 # minimum normalised score to commit
    warm_up_frames: int = 45           # frames before is_stable turns True


@dataclass
class AttentionConfig:
    ema_alpha: float = 0.10
    optimal_blink_low: float = 12.0    # blinks/min lower bound for focused range
    optimal_blink_high: float = 20.0   # blinks/min upper bound for focused range


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class RPPGConfig:
    # Physiological BPM range
    bpm_low: float = 42.0
    bpm_high: float = 180.0
    # Buffer duration in seconds for FFT stability
    buffer_seconds: float = 10.0
    # Bandpass filter cutoff in Hz
    filter_low: float = 0.7
    filter_high: float = 3.0
    filter_order: int = 4
    # Minimum signal quality for valid reading
    min_signal_quality: float = 0.3
    # ROI landmark indices (MediaPipe Face Mesh)
    forehead_landmarks: Tuple[int, ...] = (10, 338, 297, 332, 284)
    left_cheek_landmarks: Tuple[int, ...] = (205, 50, 187, 123, 116)
    right_cheek_landmarks: Tuple[int, ...] = (425, 280, 411, 352, 345)


@dataclass
class FatigueConfig:
    # EAR threshold — below this is "closed"
    ear_threshold: float = 0.22
    # Consecutive frames below EAR to count as blink
    ear_consec_frames: int = 3
    # EAR for drowsy (sustained closure)
    ear_drowsy_threshold: float = 0.19
    # MAR threshold for yawn detection
    mar_threshold: float = 0.65
    # PERCLOS window in seconds
    perclos_window: float = 30.0
    # Blink rate window in seconds
    blink_window: float = 60.0
    # Normal blink rate range (blinks/min)
    normal_blink_low: float = 10.0
    normal_blink_high: float = 25.0


@dataclass
class StressConfig:
    # HR trend window in seconds
    hr_trend_window: float = 60.0
    # Head movement smoothing window
    head_movement_window: int = 15
    # Weights for rule-based fusion
    weight_hr_variability: float = 0.35
    weight_blink_rate: float = 0.25
    weight_head_movement: float = 0.20
    weight_facial_tension: float = 0.20


@dataclass
class RiskConfig:
    # BPM thresholds
    hr_low_warning: float = 50.0
    hr_high_warning: float = 100.0
    hr_high_risk: float = 120.0
    # Fatigue thresholds
    fatigue_warning: float = 50.0
    fatigue_high_risk: float = 75.0
    # Stress thresholds
    stress_warning: float = 50.0
    stress_high_risk: float = 75.0
    # History window for trend analysis
    trend_window: int = 30


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    rppg: RPPGConfig = field(default_factory=RPPGConfig)
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dashboard_update_interval: float = 0.1
    chart_history_seconds: int = 30


# Module-wide singleton
settings = Settings()
