"""Computer-vision module (YOLOv8). Imported lazily so the rest of the app runs without torch.

1) count_people(): pre-trained COCO model -> occupancy of a room/corridor from a camera frame.
   Only the COUNT is used. No faces or images are stored (privacy by design).
2) detect_waste(): your fine-tuned weights at models/waste.pt (overflowing bins / litter).
   Train on Colab using a Roboflow/Kaggle garbage dataset (see README).
"""
import io
import os

import numpy as np
from PIL import Image

WASTE_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "waste.pt")
_cache = {}


def _model(path):
    if path not in _cache:
        from ultralytics import YOLO  # pip install ultralytics
        _cache[path] = YOLO(path)
    return _cache[path]


def _load(image_bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def count_people(image_bytes: bytes, conf: float = 0.35):
    """Returns (person_count, annotated RGB image)."""
    res = _model("yolov8n.pt")(_load(image_bytes), conf=conf, verbose=False)[0]
    n = sum(1 for b in res.boxes if int(b.cls) == 0)  # COCO class 0 = person
    return n, res.plot()[..., ::-1]


def waste_available() -> bool:
    return os.path.exists(WASTE_WEIGHTS)


def detect_waste(image_bytes: bytes, conf: float = 0.3):
    """Returns (list of (label, confidence), annotated RGB image)."""
    model = _model(WASTE_WEIGHTS)
    res = model(_load(image_bytes), conf=conf, verbose=False)[0]
    found = [(model.names[int(b.cls)], float(b.conf)) for b in res.boxes]
    return found, res.plot()[..., ::-1]
