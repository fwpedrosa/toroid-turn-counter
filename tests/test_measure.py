"""Ruler calibration: measure a synthetic toroid of known size next to the printed card."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from make_ruler import render_card  # noqa: E402

from toroid_counter import measure  # noqa: E402
from toroid_counter.core import count_turns  # noqa: E402
from toroid_counter.synth import make_toroid, tilt  # noqa: E402

PPM = 10.0  # px per mm in the fake photo


def scene(deg=0):
    tor = make_toroid(40, size=1200, seed=1, lead=False, glare=False)  # r_out 480 px -> 96 mm
    card = cv2.cvtColor(render_card(PPM), cv2.COLOR_GRAY2BGR)
    H, W = 1200 + card.shape[0] + 60, max(1200, card.shape[1]) + 40
    img = np.full((H, W, 3), 25, np.uint8)
    img[:1200, 20:1220] = tor
    img[1230:1230 + card.shape[0], 20:20 + card.shape[1]] = card
    return tilt(img, deg, 20) if deg else img


@pytest.mark.parametrize("deg", [0, 20])
def test_measure_wound_od_id(deg):
    img = scene(deg)
    res = count_turns(img)
    m = measure.measure(img, res.ring, res.count)
    assert m and m["markers"] == 2
    # synthetic winding spans 0.97*r_in .. 1.03*r_out: OD ~ 96-99 mm, ID ~ 42-44 mm
    assert 94 < m["od_mm"] < 101, m
    assert 40 < m["id_mm"] < 46, m


def test_no_card():
    img = make_toroid(30, seed=2)
    assert measure.measure(img, count_turns(img).ring, 30) is None
