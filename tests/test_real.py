"""Real captures from the bench (ground truth counted by hand). Add more as they come in:
name them <what>_<turns>t.png in tests/real/."""
import re
from pathlib import Path

import cv2
import pytest

from toroid_counter.core import count_turns

REAL = sorted(p for p in (Path(__file__).parent / "real").glob("*.png") if "screenshot" not in p.name)


@pytest.mark.parametrize("path", REAL, ids=[p.stem for p in REAL])
def test_real_capture(path):
    truth = int(re.search(r"_(\d+)t", path.stem).group(1))
    res = count_turns(cv2.imread(str(path)))
    assert abs(res.count - truth) <= 1, res.summary()
