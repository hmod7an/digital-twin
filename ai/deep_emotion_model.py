"""
Deep emotion model — ensemble of two AffectNet EfficientNet-B0 ONNX models.

  vgaf: trained on VGGFace2 + AffectNet  (general expressions)
  afew: trained on AFEW video dataset     (acted/dynamic, more anger-sensitive)

Predictions are averaged with equal weight then mapped to our 9-state
vocabulary.  Inference runs every STRIDE frames (~10 fps) and is cached
for intermediate frames.
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

_MODEL_DIR = os.path.expanduser("~/.hsemotion")

# (filename, class_list, weight)
_MODELS: List[Tuple[str, List[str], float]] = [
    ("enet_b0_8_best_vgaf.onnx",
     ["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"],
     1.0),
    ("enet_b0_8_best_afew.onnx",
     ["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"],
     1.0),
]

_AFFECTNET_TO_OURS: Dict[str, Dict[str, float]] = {
    "Anger":     {"Angry": 0.80, "Stressed": 0.20},
    "Contempt":  {"Angry": 0.40, "Neutral":  0.60},
    "Disgust":   {"Angry": 0.50, "Stressed": 0.50},
    "Fear":      {"Stressed": 0.75, "Sad": 0.25},
    "Happiness": {"Happy": 1.00},
    "Neutral":   {"Neutral": 1.00},
    "Sadness":   {"Sad": 1.00},
    "Surprise":  {"Surprised": 1.00},
}

_OUR_STATES = [
    "Neutral", "Happy", "Sad", "Angry",
    "Stressed", "Tired", "Surprised", "Focused", "Distracted",
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepEmotionModel:
    """
    Ensemble of two ONNX emotion models for improved accuracy.
    Thread-safe when called from a single background thread.
    """

    STRIDE = 3  # run every N frames (~10 fps on average)

    def __init__(self):
        self._sessions: List = []      # list of (sess, input_name, classes, weight)
        self._frame_count: int = 0
        self._cached: Optional[Dict[str, float]] = None
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
        except ImportError:
            return

        for fname, classes, weight in _MODELS:
            path = os.path.join(_MODEL_DIR, fname)
            if not os.path.exists(path):
                continue
            try:
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                inp  = sess.get_inputs()[0].name
                self._sessions.append((sess, inp, classes, weight))
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return len(self._sessions) > 0

    def predict(self, face_bgr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """
        face_bgr — HxWx3 BGR crop of the face.
        Returns normalised probability dict over our 9 states, or None.
        Runs inference every STRIDE frames; returns cached result otherwise.
        """
        self._frame_count += 1
        if not self.available or face_bgr is None or face_bgr.size == 0:
            return self._cached
        if self._frame_count % self.STRIDE != 0:
            return self._cached

        try:
            x = self._preprocess(face_bgr)
            accumulated = np.zeros(len(_AFFECTNET_TO_OURS), dtype=np.float32)
            total_weight = 0.0

            for sess, inp_name, classes, weight in self._sessions:
                logits = sess.run(None, {inp_name: x})[0][0]
                probs  = np.exp(logits - logits.max())
                probs /= probs.sum()

                # Map this model's class probs to AffectNet-8 slot order
                mapped = np.zeros(8, dtype=np.float32)
                affectnet_order = ["Anger","Contempt","Disgust","Fear",
                                   "Happiness","Neutral","Sadness","Surprise"]
                for j, cls in enumerate(classes):
                    if cls in affectnet_order:
                        mapped[affectnet_order.index(cls)] += float(probs[j])

                accumulated[:len(mapped)] += mapped * weight
                total_weight += weight

            if total_weight > 0:
                accumulated /= total_weight

            affectnet_classes = ["Anger","Contempt","Disgust","Fear",
                                  "Happiness","Neutral","Sadness","Surprise"]
            scores: Dict[str, float] = {s: 0.0 for s in _OUR_STATES}
            for i, cls in enumerate(affectnet_classes):
                for target, w in _AFFECTNET_TO_OURS[cls].items():
                    scores[target] += float(accumulated[i]) * w

            total = sum(scores.values()) + 1e-9
            self._cached = {s: v / total for s, v in scores.items()}

        except Exception:
            pass

        return self._cached

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(face_bgr[:, :, ::-1])
        img = img.resize((224, 224), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
        return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
