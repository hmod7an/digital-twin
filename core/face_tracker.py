"""
MediaPipe Face Landmarker wrapper (Tasks API — MediaPipe ≥ 0.10.14).

Uses FaceLandmarker (Tasks API) which replaced the deprecated
mp.solutions.face_mesh in MediaPipe 0.10.x.

References:
  Kartynnik et al. (2019) — Real-time Facial Surface Geometry from Monocular Video
  arXiv:1907.06724
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass, field
from typing import Optional, Tuple

from config.settings import settings
from core.model_manager import ensure_face_landmarker


@dataclass
class FaceLandmarks:
    """Processed landmark data for a single detected face."""
    raw: np.ndarray                  # (478, 3) normalised x,y,z
    pixel_coords: np.ndarray         # (478, 2) pixel x,y
    forehead_roi: Optional[tuple] = None   # (B, G, R) mean
    left_cheek_roi: Optional[tuple] = None
    right_cheek_roi: Optional[tuple] = None
    face_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x,y,w,h
    head_pose_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw,pitch,roll


class FaceTracker:
    """
    Runs MediaPipe FaceLandmarker (Tasks API) on each frame and extracts:
    - 478 3-D landmark coordinates
    - Forehead and cheek ROI mean colours for rPPG
    - Head pose angles for stress estimation
    """

    # Eye landmarks (EAR)
    LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Mouth landmarks (MAR)
    MOUTH_INDICES = [61, 291, 13, 14, 78, 308, 95, 324]
    # Nose tip
    NOSE_TIP = 1

    # Face oval indices for drawing
    _OVAL_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109, 10,
    ]

    def __init__(self):
        model_path = ensure_face_landmarker()

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Run face landmarker on a BGR frame.
        Returns FaceLandmarks or None if no face detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm_list = result.face_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
        pixel_coords = (raw[:, :2] * np.array([w, h])).astype(int)

        landmarks = FaceLandmarks(raw=raw, pixel_coords=pixel_coords)
        landmarks.forehead_roi = self._extract_roi(
            frame, pixel_coords, settings.rppg.forehead_landmarks
        )
        landmarks.left_cheek_roi = self._extract_roi(
            frame, pixel_coords, settings.rppg.left_cheek_landmarks
        )
        landmarks.right_cheek_roi = self._extract_roi(
            frame, pixel_coords, settings.rppg.right_cheek_landmarks
        )
        landmarks.face_bbox = self._face_bounding_box(pixel_coords, w, h)
        landmarks.head_pose_angles = self._estimate_head_pose(raw, w, h)
        return landmarks

    def draw_landmarks(self, frame: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Draw face oval and ROI polygons on frame."""
        annotated = frame.copy()
        coords = landmarks.pixel_coords

        # Face oval
        for idx in self._OVAL_INDICES:
            if idx < len(coords):
                cv2.circle(annotated, tuple(coords[idx]), 1, (0, 212, 255), -1)

        # Connect oval points
        oval_pts = [coords[i] for i in self._OVAL_INDICES if i < len(coords)]
        for i in range(1, len(oval_pts)):
            cv2.line(annotated, tuple(oval_pts[i-1]), tuple(oval_pts[i]),
                     (0, 212, 255), 1)

        # ROI highlight boxes
        for roi_indices, color in [
            (settings.rppg.forehead_landmarks, (0, 255, 100)),
            (settings.rppg.left_cheek_landmarks, (255, 120, 0)),
            (settings.rppg.right_cheek_landmarks, (255, 120, 0)),
        ]:
            pts = np.array([coords[i] for i in roi_indices
                            if i < len(coords)], dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(annotated, [pts], True, color, 2)

        return annotated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_roi(
        frame: np.ndarray,
        pixel_coords: np.ndarray,
        indices: Tuple[int, ...],
    ) -> Optional[tuple]:
        """Extract mean BGR colour inside a polygon ROI."""
        valid = [i for i in indices if i < len(pixel_coords)]
        if len(valid) < 3:
            return None
        pts = np.array([pixel_coords[i] for i in valid], dtype=np.int32)
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        if mask.sum() == 0:
            return None
        mean = cv2.mean(frame, mask=mask)
        return (mean[0], mean[1], mean[2])  # B, G, R

    @staticmethod
    def _face_bounding_box(
        pixel_coords: np.ndarray, w: int, h: int
    ) -> Tuple[int, int, int, int]:
        x_min = int(max(0, pixel_coords[:, 0].min()))
        y_min = int(max(0, pixel_coords[:, 1].min()))
        x_max = int(min(w, pixel_coords[:, 0].max()))
        y_max = int(min(h, pixel_coords[:, 1].max()))
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    @staticmethod
    def _estimate_head_pose(
        raw: np.ndarray, w: int, h: int
    ) -> Tuple[float, float, float]:
        """
        Lightweight head-pose estimate via solvePnP using 6 anchor points.
        Returns (yaw, pitch, roll) in degrees.
        """
        model_points = np.array([
            (0.0,    0.0,    0.0),       # Nose tip (1)
            (0.0,  -330.0,  -65.0),      # Chin (152)
            (-225.0, 170.0, -135.0),     # Left eye corner (33)
            (225.0,  170.0, -135.0),     # Right eye corner (263)
            (-150.0, -150.0, -125.0),    # Left mouth corner (61)
            (150.0,  -150.0, -125.0),    # Right mouth corner (291)
        ], dtype=np.float64)

        anchor_indices = [1, 152, 33, 263, 61, 291]
        if max(anchor_indices) >= len(raw):
            return (0.0, 0.0, 0.0)

        image_points = np.array(
            [(raw[i][0] * w, raw[i][1] * h) for i in anchor_indices],
            dtype=np.float64,
        )

        focal = float(w)
        camera_matrix = np.array([
            [focal, 0,     w / 2.0],
            [0,     focal, h / 2.0],
            [0,     0,     1.0    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success:
            return (0.0, 0.0, 0.0)

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
            yaw   = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
            roll  = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
            yaw   = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
            roll  = 0.0
        return (yaw, pitch, roll)

    def close(self):
        self._detector.close()
