"""
Facial geometry feature extraction from MediaPipe Face Mesh landmarks.

Features (all normalised, scale-invariant, adaptive to face geometry):
  - Mouth curve:         smile / frown via corner Y vs lip-centre Y
  - Brow furrow:         adaptive inner-brow ratio (person-calibrated)
  - Brow raise:          apex height vs adaptive personal baseline
  - Inner brow raise:    AU1 — inner corner lift, primary sadness cue
  - Mouth opening:       vertical / horizontal lip ratio
  - Eye openness:        Eye Aspect Ratio (EAR)
  - Head motion:         RMS frame-to-frame yaw/pitch diff (rolling)
  - Facial energy:       rolling expression intensity + head motion
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.face_tracker import FaceLandmarks


@dataclass
class FaceFeatures:
    """Scale-invariant facial geometry features for emotion/attention inference."""

    # ── Mouth ────────────────────────────────────────────────────────
    smile_score: float = 0.0        # 0=neutral/frown,  1=strong smile
    frown_score: float = 0.0        # 0=neutral/smile,  1=strong frown (corner droop)
    mouth_open_ratio: float = 0.0   # 0=closed,         1=wide open (MAR proxy)
    mouth_width_norm: float = 0.0   # mouth width / face width

    # ── Brows ────────────────────────────────────────────────────────
    brow_furrow_score: float = 0.0      # 0=relaxed, 1=inner brows maximally close
    brow_raise_score: float = 0.0       # 0=baseline, 1=fully raised (whole brow)
    inner_brow_raise_score: float = 0.0 # AU1: inner corner lift — key sadness signal

    # ── Eyes ─────────────────────────────────────────────────────────
    ear: float = 0.30               # Eye Aspect Ratio
    eye_wide_score: float = 0.0     # 0=normal, 1=wide-open (surprise/alert)

    # ── Head ─────────────────────────────────────────────────────────
    head_yaw: float = 0.0           # degrees, positive = right turn
    head_pitch: float = 0.0         # degrees, positive = upward tilt
    head_yaw_norm: float = 0.0      # |yaw|/45°, clipped 0-1
    head_movement_score: float = 0.0  # 0=still, 1=very active

    # ── Overall facial activation ─────────────────────────────────────
    facial_energy: float = 0.5      # 0=very passive (sad/tired), 1=very active

    valid: bool = True


_FALLBACK = FaceFeatures(valid=False)

# EAR landmark indices (MediaPipe Face Mesh 478 total)
_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class FeatureExtractor:
    """
    Converts a FaceLandmarks object into a normalised FaceFeatures snapshot.

    All thresholds adapt to the individual's face geometry via rolling-median
    calibration buffers. No hard-coded geometry assumptions remain after the
    first 30 frames of calibration.
    """

    # ── Landmark indices ─────────────────────────────────────────────
    _MOUTH_L    = 61    # left mouth corner
    _MOUTH_R    = 291   # right mouth corner
    _LIP_TOP    = 13    # upper lip centre (outer surface)
    _LIP_BOT    = 14    # lower lip centre (outer surface)
    _FOREHEAD   = 10    # mid-forehead (face-height scale)
    _CHIN       = 152   # chin centre
    _L_IBROW    = 55    # left inner eyebrow tip
    _R_IBROW    = 285   # right inner eyebrow tip
    _L_BROW_AP  = 63    # left brow apex (highest point)
    _R_BROW_AP  = 293   # right brow apex
    _L_EYE_CTR  = 159   # left eye centre proxy
    _R_EYE_CTR  = 386   # right eye centre proxy

    _EAR_NORMAL = 0.30
    _EAR_WIDE   = 0.37  # above → wide-open / surprised

    _MOTION_WIN    = 20   # frames for head-motion RMS window
    _CAL_FRAMES    = 30   # frames needed to set adaptive baselines
    _ENERGY_WIN    = 45   # frames for facial-energy rolling window

    def __init__(self):
        self._yaw_buf   = deque(maxlen=self._MOTION_WIN)
        self._pitch_buf = deque(maxlen=self._MOTION_WIN)

        # Adaptive baselines — calibrate to this person's resting geometry
        self._brow_cal_buf: deque    = deque(maxlen=120)
        self._brow_baseline: Optional[float] = None

        self._furrow_cal_buf: deque  = deque(maxlen=120)
        self._furrow_baseline: Optional[float] = None

        self._ibrow_cal_buf: deque   = deque(maxlen=120)
        self._ibrow_baseline: Optional[float] = None

        # Adaptive mouth-curve baseline — person's natural corner position.
        # Without this, naturally upturned corners read as permanent smile.
        self._mouth_cal_buf: deque   = deque(maxlen=120)
        self._mouth_baseline: Optional[float] = None   # raw (lip_mid_y - corner_y)

        # Rolling expression intensity for facial-energy computation
        self._energy_buf: deque      = deque(maxlen=self._ENERGY_WIN)

    # ── Public interface ─────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return (self._brow_baseline is not None
                and self._furrow_baseline is not None
                and self._ibrow_baseline is not None)

    def extract(self, landmarks: Optional[FaceLandmarks]) -> FaceFeatures:
        if landmarks is None:
            return _FALLBACK
        try:
            return self._extract(landmarks)
        except Exception:
            return _FALLBACK

    # ── Internal extraction ──────────────────────────────────────────

    def _extract(self, lm: FaceLandmarks) -> FaceFeatures:
        c = lm.pixel_coords.astype(float)   # (478, 2) pixel coords
        yaw, pitch, _ = lm.head_pose_angles

        # Face-height scale reference
        face_h = float(np.linalg.norm(c[self._FOREHEAD] - c[self._CHIN]))
        if face_h < 20:
            return _FALLBACK

        # ── Mouth curve (adaptive to personal resting geometry) ──────
        # Y increases downward. Positive raw = corners above lip centre (smile).
        # Calibrate to this person's natural resting mouth curve so that a
        # naturally upturned face reads as 0, not a permanent smile.
        corner_y  = (c[self._MOUTH_L][1] + c[self._MOUTH_R][1]) / 2.0
        lip_mid_y = (c[self._LIP_TOP][1]  + c[self._LIP_BOT][1])  / 2.0
        raw_curve = lip_mid_y - corner_y   # positive = smile-direction

        self._mouth_cal_buf.append(raw_curve)
        if self._mouth_baseline is None and len(self._mouth_cal_buf) >= self._CAL_FRAMES:
            self._mouth_baseline = float(np.median(self._mouth_cal_buf))
        mb = self._mouth_baseline if self._mouth_baseline is not None else 0.0

        # Relative curve: how much MORE or LESS than resting position.
        rel_curve = raw_curve - mb
        smile_sc  = float(np.clip( rel_curve / (face_h * 0.08 + 1e-9), 0.0, 1.0))
        frown_sc  = float(np.clip(-rel_curve / (face_h * 0.06 + 1e-9), 0.0, 1.0))

        # ── Mouth openness (MAR) ─────────────────────────────────────
        vert      = float(np.linalg.norm(c[self._LIP_TOP] - c[self._LIP_BOT]))
        horiz     = float(np.linalg.norm(c[self._MOUTH_L] - c[self._MOUTH_R])) + 1e-9
        mouth_open = float(np.clip(vert / horiz, 0.0, 1.0))
        mw_norm    = float(np.clip(horiz / (face_h * 0.80 + 1e-9), 0.0, 1.0))

        # ── Brow furrow (adaptive) ───────────────────────────────────
        ibrow_d   = float(np.linalg.norm(c[self._L_IBROW] - c[self._R_IBROW]))
        ipupil_d  = float(np.linalg.norm(c[self._L_EYE_CTR] - c[self._R_EYE_CTR])) + 1e-9
        brow_ratio = ibrow_d / ipupil_d
        self._furrow_cal_buf.append(brow_ratio)
        if self._furrow_baseline is None and len(self._furrow_cal_buf) >= self._CAL_FRAMES:
            self._furrow_baseline = float(np.median(self._furrow_cal_buf))
        fb = self._furrow_baseline if self._furrow_baseline is not None else 0.70
        furrow_sc = float(np.clip((fb - 0.05 - brow_ratio) / 0.20, 0.0, 1.0))

        # ── Brow raise (adaptive) ────────────────────────────────────
        l_rise  = float(c[self._L_EYE_CTR][1] - c[self._L_BROW_AP][1])
        r_rise  = float(c[self._R_EYE_CTR][1] - c[self._R_BROW_AP][1])
        avg_rise = (l_rise + r_rise) / 2.0
        self._brow_cal_buf.append(avg_rise)
        if self._brow_baseline is None and len(self._brow_cal_buf) >= self._CAL_FRAMES:
            self._brow_baseline = float(np.median(self._brow_cal_buf))
        bl = self._brow_baseline if self._brow_baseline is not None else avg_rise
        raise_sc = float(np.clip((avg_rise - bl) / (face_h * 0.12 + 1e-9), 0.0, 1.0))

        # ── Inner brow raise — AU1 (sadness cue) ────────────────────
        # Inner brow tips move UP when sad; Y decreases = rise.
        ibrow_y = (c[self._L_IBROW][1] + c[self._R_IBROW][1]) / 2.0
        self._ibrow_cal_buf.append(ibrow_y)
        if self._ibrow_baseline is None and len(self._ibrow_cal_buf) >= self._CAL_FRAMES:
            self._ibrow_baseline = float(np.median(self._ibrow_cal_buf))
        ib = self._ibrow_baseline if self._ibrow_baseline is not None else ibrow_y
        inner_brow_raise_sc = float(np.clip((ib - ibrow_y) / (face_h * 0.06 + 1e-9), 0.0, 1.0))

        # ── Eye Aspect Ratio ─────────────────────────────────────────
        ear = _compute_ear(c)
        eye_wide = float(np.clip((ear - self._EAR_WIDE) / 0.07, 0.0, 1.0))

        # ── Head pose & motion ────────────────────────────────────────
        self._yaw_buf.append(yaw)
        self._pitch_buf.append(pitch)
        yaw_norm = float(np.clip(abs(yaw) / 45.0, 0.0, 1.0))

        motion = 0.0
        if len(self._yaw_buf) >= 4:
            dy  = np.diff(np.array(self._yaw_buf))
            dp  = np.diff(np.array(self._pitch_buf))
            motion = float(np.clip(np.sqrt(np.mean(dy**2 + dp**2)) / 8.0, 0.0, 1.0))

        # ── Facial energy ─────────────────────────────────────────────
        # Combines: rolling expression intensity + head motion + eye alertness.
        # Low = passive / sad / tired.  High = active / expressive / engaged.
        expr_intensity = max(smile_sc, frown_sc, furrow_sc, raise_sc,
                             eye_wide, inner_brow_raise_sc, mouth_open * 0.5)
        self._energy_buf.append(expr_intensity)
        expr_mean  = float(np.mean(self._energy_buf)) if self._energy_buf else 0.3
        ear_alert  = float(np.clip((ear - 0.22) / 0.14, 0.0, 1.0))
        facial_energy = float(np.clip(
            0.40 * motion + 0.40 * expr_mean + 0.20 * ear_alert,
            0.0, 1.0
        ))

        return FaceFeatures(
            smile_score=smile_sc,
            frown_score=frown_sc,
            mouth_open_ratio=mouth_open,
            mouth_width_norm=mw_norm,
            brow_furrow_score=furrow_sc,
            brow_raise_score=raise_sc,
            inner_brow_raise_score=inner_brow_raise_sc,
            ear=ear,
            eye_wide_score=eye_wide,
            head_yaw=float(yaw),
            head_pitch=float(pitch),
            head_yaw_norm=yaw_norm,
            head_movement_score=motion,
            facial_energy=facial_energy,
            valid=True,
        )


def _compute_ear(c: np.ndarray) -> float:
    def _ear(idx):
        try:
            p = [c[i] for i in idx]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h  = np.linalg.norm(p[0] - p[3])
            return float((v1 + v2) / (2.0 * h + 1e-9))
        except Exception:
            return 0.30
    return (_ear(_LEFT_EYE) + _ear(_RIGHT_EYE)) / 2.0
