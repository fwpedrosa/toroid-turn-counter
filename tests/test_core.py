"""Accuracy tests on synthetic toroids with known turn counts. Run: python -m pytest -q"""
import pytest

from toroid_counter.core import CountParams, count_turns, find_ring
from toroid_counter.synth import make_toroid

CASES = [
    (10, {}), (25, {}), (50, {}), (80, {}), (120, {}),
    (200, {"size": 1600}), (300, {"size": 2000}),
    (40, {"dark_bg": False}), (35, {"wire_fill": 0.6}), (150, {"wire_fill": 1.0}),
    (70, {"offset": (80, -60)}), (100, {"jitter": 0.15}),
]


@pytest.mark.parametrize("turns,kw", CASES)
def test_count_within_one(turns, kw):
    img = make_toroid(turns, seed=turns, **kw)
    res = count_turns(img)
    assert abs(res.count - turns) <= 1, res.summary()


def test_ring_found_near_truth():
    img = make_toroid(60, size=1200, seed=1)
    ring = find_ring(img)
    assert abs(ring.cx - 600) < 15 and abs(ring.cy - 600) < 15
    assert abs(ring.r_out - 480) < 30 and abs(ring.r_in - 216) < 20


def test_partial_winding_is_flagged():
    res = count_turns(make_toroid(60, gap_deg=40, seed=8))
    assert abs(res.count - 60) <= 1
    assert any("gap" in f for f in res.flags)


def test_multilayer_warning():
    res = count_turns(make_toroid(50, seed=2), params=CountParams(expected_layers=2))
    assert any("MULTI-LAYER" in f for f in res.flags)


def test_too_many_for_single_layer():
    # T50 (ID 7.7 mm) with 1.08 mm wire holds at most ~19 turns in one layer
    res = count_turns(make_toroid(50, seed=2), params=CountParams(toroid_id_mm=7.7, wire_od_mm=1.08))
    assert any("max single layer" in f for f in res.flags)


def test_low_resolution_flag():
    res = count_turns(make_toroid(150, size=500, seed=4))
    assert any("LOW RESOLUTION" in f for f in res.flags) or abs(res.count - 150) <= 2


@pytest.mark.parametrize("deg", [15, 25, 35])
@pytest.mark.parametrize("turns", [24, 60, 120])
def test_tilted_camera_oval(turns, deg):
    from toroid_counter.synth import tilt
    img = tilt(make_toroid(turns, seed=turns + deg), deg, direction_deg=3 * deg)
    res = count_turns(img)
    assert abs(res.count - turns) <= 1, res.summary()
