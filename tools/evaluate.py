"""Score the counter on a folder of labelled images.

Label = file name containing '<turns>t', e.g. T68_AWG24_052t_cam1.jpg  (truth = 52 turns).
Usage: python tools/evaluate.py images_folder   -> prints per-image result and accuracy, writes eval.csv
"""
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toroid_counter.core import count_turns  # noqa: E402

folder = Path(sys.argv[1])
rows = []
for p in sorted(folder.iterdir()):
    m = re.search(r"(\d+)t", p.stem)
    if not m or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        continue
    truth = int(m.group(1))
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    try:
        r = count_turns(img)
        rows.append((p.name, truth, r.count, r.count - truth, round(r.confidence, 2), " | ".join(r.flags)))
    except Exception as e:
        rows.append((p.name, truth, -1, None, 0, f"ERROR {e}"))
    print(*rows[-1], sep="\t")
ok = [r for r in rows if r[3] == 0]
near = [r for r in rows if r[3] is not None and abs(r[3]) <= 1]
flagged_wrong = [r for r in rows if r[3] not in (0,) and r[5]]
wrong = [r for r in rows if r[3] != 0]
print(f"\n{len(rows)} images | exact {len(ok)} ({len(ok)/max(len(rows),1):.0%}) | within 1: {len(near)}"
      f" | wrong but flagged: {len(flagged_wrong)}/{len(wrong)}")
with open(folder / "eval.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "truth", "count", "error", "confidence", "flags"])
    w.writerows(rows)
