"""Real captures from the bench (ground truth counted by hand). Add more as they come in:
name them <what>_<turns>t.png in tests/real/."""
import re
from pathlib import Path

import cv2
import pytest

from toroid_counter.core import count_turns

REAL = sorted(p for p in (Path(__file__).parent / "real").glob("*.png")
              if "screenshot" not in p.name and "noring" not in p.name)


@pytest.mark.parametrize("path", REAL, ids=[p.stem for p in REAL])
def test_real_capture(path):
    truth = int(re.search(r"_(\d+)t", path.stem).group(1))
    res = count_turns(cv2.imread(str(path)))
    assert abs(res.count - truth) <= 1, res.summary()


NORING = sorted((Path(__file__).parent / "real").glob("*_noring.png"))


@pytest.mark.parametrize("path", NORING, ids=[p.stem for p in NORING])
def test_no_false_ring(path):
    """Hand-held live frames where the old finder locked onto a dark shirt: the finder must not
    return a huge ring running off the picture (it should ask the operator instead)."""
    from toroid_counter.core import find_ring
    img = cv2.imread(str(path))
    ring = find_ring(img)
    if ring is not None:
        h, w = img.shape[:2]
        assert ring.r_out < 0.6 * min(h, w)
