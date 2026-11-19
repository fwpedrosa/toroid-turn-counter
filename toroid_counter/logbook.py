"""Append every confirmed result to a CSV and keep the image + overlay next to it."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import cv2

FIELDS = ["timestamp", "part", "operator", "toroid", "wire", "layers", "expected", "recipe", "sections", "auto_count", "od_mm", "id_mm", "inner_pitch_mm",
          "final_count", "added", "removed", "result", "confidence", "flags", "ring", "image", "overlay"]


def default_dir() -> Path:
    return Path.home() / "Documents" / "ToroidCounter"


def log_result(base: Path, image, overlay, row: dict) -> Path:
    now = dt.datetime.now()
    day = base / now.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%H%M%S") + "_" + "".join(ch for ch in (row.get("part") or "part") if ch.isalnum() or ch in "-_")
    img_p, ov_p = day / f"{stem}.png", day / f"{stem}_overlay.jpg"
    cv2.imwrite(str(img_p), image)
    cv2.imwrite(str(ov_p), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    row = dict(row, timestamp=now.isoformat(timespec="seconds"), image=str(img_p), overlay=str(ov_p))
    if isinstance(row.get("ring"), dict):
        row["ring"] = json.dumps({k: round(v, 1) for k, v in row["ring"].items()})
    if isinstance(row.get("flags"), list):
        row["flags"] = " | ".join(row["flags"])
    csv_p = base / "results.csv"
    new = not csv_p.exists()
    with open(csv_p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    return csv_p


def save_sample(base: Path, image, ring, angles, row: dict) -> Path:
    """Keep every confirmed result as a labelled training sample (learning from the operator).

    dataset/images/<id>.png   raw image
    dataset/labels/<id>.txt   YOLO format, class 0 = one box per turn (operator-corrected marks)
    dataset/annotations/<id>.json  ring, turn angles, counts, part data (for re-tuning OpenCV)
    """
    import math

    ds = base / "dataset"
    for sub in ("images", "labels", "annotations"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    stem = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = image.shape[:2]
    cv2.imwrite(str(ds / "images" / f"{stem}.png"), image)
    angles = sorted(angles)
    pitch = 2 * math.pi * (ring.r_in + 0.5 * (ring.r_out - ring.r_in)) / max(len(angles), 1)
    lines = []
    for a in angles:
        pts = [ring.pt(a, ring.r_in + f * (ring.r_out - ring.r_in)) for f in (0.0, 1.0)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = max(3.0, 0.45 * pitch)
        x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
        y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    (ds / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ann = {"ring": ring.to_dict(), "angles": angles, "count": len(angles),
           **{k: v for k, v in row.items() if isinstance(v, (str, int, float))}}
    (ds / "annotations" / f"{stem}.json").write_text(json.dumps(ann, indent=1), encoding="utf-8")
    yaml = ds / "data.yaml"
    if not yaml.exists():
        yaml.write_text(f"path: {ds.as_posix()}\ntrain: images\nval: images\nnames:\n  0: turn\n", encoding="utf-8")
    return ds
