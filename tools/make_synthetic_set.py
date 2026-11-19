"""Write synthetic toroid images with known counts, e.g. to try the app without a microscope.
Usage: python tools/make_synthetic_set.py out_folder [n_images]
File name holds the truth: toroid_<turns>t_<index>.png
"""
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toroid_counter.synth import make_toroid  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
for i in range(n):
    turns = rng.randint(8, 250)
    img = make_toroid(turns, size=1600, seed=i, wire_fill=rng.uniform(0.6, 1.0),
                      jitter=rng.uniform(0, 0.12), dark_bg=rng.random() < 0.7,
                      gap_deg=rng.choice([0, 0, 0, 30]))
    cv2.imwrite(str(out / f"toroid_{turns}t_{i:03d}.png"), img)
print(f"wrote {n} images to {out}")
