"""Optional YOLO detector for hard images (Ultralytics, AGPL-3.0).

Not included in the default .exe (PyTorch makes it ~1 GB). Install with
`pip install ultralytics` and put a trained model at models/turns.pt.
The model detects one class, "turn": a box around each wire crossing the top face.
Detections are turned into angles around the ring so the rest of the app
(overlay, manual correction, log) works the same as with the OpenCV counter.
"""
from __future__ import annotations

import math
from pathlib import Path

_model = None


def model_path() -> Path:
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "models" / "turns.pt"


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return model_path().exists()


def detect_angles(img, ring, conf: float = 0.35) -> list[float]:
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(model_path()))
    res = _model.predict(img, conf=conf, verbose=False)[0]
    angles = []
    for x0, y0, x1, y1 in res.boxes.xyxy.tolist():
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        a, r = ring.polar_of(x, y)
        if ring.r_in * 0.9 <= r <= ring.r_out * 1.1:
            angles.append(a)
    angles.sort()
    # merge duplicates closer than a third of the median spacing
    if len(angles) > 2:
        gaps = [b - a for a, b in zip(angles, angles[1:])]
        med = sorted(gaps)[len(gaps) // 2]
        merged = [angles[0]]
        for a in angles[1:]:
            if a - merged[-1] > med / 3:
                merged.append(a)
        angles = merged
    return angles
