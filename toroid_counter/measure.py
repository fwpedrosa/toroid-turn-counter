"""Real-size measurements with the printed calibration ruler (tools/make_ruler.py).

The ruler card carries two ArUco markers (DICT_4X4_50, ids 0 and 1, 15 mm squares) at known
positions. Their 8 corners give a homography from image pixels to millimetres on the card plane,
which also undoes a moderate camera tilt. Put the card next to the toroid, with its surface at the
same height as the top of the winding (use the printed riser), or sizes come out a few % off.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MARKER_MM = 15.0
# top-left corner of each marker on the card, in mm (card origin = top-left of the card)
MARKER_TL = {0: (3.0, 9.5), 1: (155.0, 9.5)}
CARD_W, CARD_H = 173.0, 34.0
RULER_X0, RULER_LEN = 23.0, 127.0  # 5 in

_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_corners_mm(mid: int) -> np.ndarray:
    x, y = MARKER_TL[mid]
    s = MARKER_MM
    return np.array([(x, y), (x + s, y), (x + s, y + s), (x, y + s)], np.float32)  # TL TR BR BL


def find_card(img: np.ndarray):
    """Return (H image->mm, number of markers found, mean marker side in px) or (None, 0, 0)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None, 0, 0.0
    src, dst, sides = [], [], []
    for c, i in zip(corners, ids.ravel()):
        if int(i) in MARKER_TL:
            pts = c.reshape(4, 2)
            src.append(pts)
            dst.append(marker_corners_mm(int(i)))
            sides += [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
    if not src:
        return None, 0, 0.0
    src, dst = np.vstack(src), np.vstack(dst)
    H, _ = cv2.findHomography(src, dst, 0)
    return H, len(src) // 4, float(np.mean(sides))


def to_mm(H, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def _diameter(H, ring, r) -> float:
    a = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    mm = to_mm(H, [ring.pt(x, r) for x in a])
    # fit a circle by least squares (handles any leftover perspective)
    A = np.c_[2 * mm, np.ones(len(mm))]
    b = (mm ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return 2 * math.sqrt(c + cx * cx + cy * cy)


def measure(img, ring, turns: int):
    """Wound OD / ID in mm and the wire pitch at the inner edge. None when no card is visible."""
    H, n, side_px = find_card(img)
    if H is None:
        return None
    od = _diameter(H, ring, ring.r_out)
    idm = _diameter(H, ring, ring.r_in)
    res = {"markers": n, "od_mm": od, "id_mm": idm, "px_per_mm": side_px / MARKER_MM}
    if turns:
        # turns touch at the inner edge: pitch there ~ wire diameter (an upper bound if spaced)
        res["inner_pitch_mm"] = math.pi * idm / turns
    notes = []
    if n < 2:
        notes.append("only one marker seen - keep both markers in view for best accuracy")
    if res["px_per_mm"] < 8:
        notes.append(f"low resolution {res['px_per_mm']:.1f} px/mm - zoom in for +/-0.1 mm")
    res["notes"] = notes
    return res


def summary(m) -> str:
    if not m:
        return "No ruler in view"
    s = f"Wound OD {m['od_mm']:.2f} mm   ID {m['id_mm']:.2f} mm"
    if "inner_pitch_mm" in m:
        s += f"\nInner pitch {m['inner_pitch_mm']:.3f} mm (= wire Ø if turns touch)"
    for n in m["notes"]:
        s += "\n! " + n
    return s
