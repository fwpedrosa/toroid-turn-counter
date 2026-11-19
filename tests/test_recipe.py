"""Recipe tests: sections and wire colours on synthetic toroids."""
import pytest

from toroid_counter import recipe
from toroid_counter.core import CountParams, count_turns
from toroid_counter.synth import make_sectioned

CASES = [
    ("copper 30", [{"copper": 30}]),
    ("A: copper 20 ; B: copper 20", [{"copper": 20}] * 2),
    ("copper 18; copper 18; copper 18", [{"copper": 18}] * 3),
    ("A: copper 10 ; B: red 13 + green 13", [{"copper": 10}, {"red": 13, "green": 13}]),
    ("A: red 9+green 9; B: red 9+green 9; C: red 9 + green 9", [{"red": 9, "green": 9}] * 3),
]


def test_parse():
    secs = recipe.parse("A: copper 10 ; B: red 13 + green 13")
    assert [s.name for s in secs] == ["A", "B"]
    assert secs[1].turns == {"red": 13, "green": 13}
    assert [s.name for s in recipe.parse("copper 18; copper 18; copper 18")] == ["A", "B", "C"]
    with pytest.raises(ValueError):
        recipe.parse("A: thirteen red")


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("text,secs", CASES)
def test_recipe_pass_or_flagged(text, secs, seed):
    """Every winding must PASS, or the error must be a visible 'unknown colour' mark (glare)."""
    img, truth = make_sectioned(secs, seed=seed + 7)
    rec = recipe.parse(text)
    res = count_turns(img, params=CountParams(sections=len(rec)))
    lines, colors, _ = recipe.evaluate(img, res.ring, res.angles, rec)
    bad = [ln for ln in lines if not ln.ok]
    assert all(ln.section == "?" for ln in bad), [(ln.section, ln.color, ln.expected, ln.counted) for ln in bad]


def test_wrong_turn_count_fails():
    img, _ = make_sectioned([{"copper": 18}] * 3, seed=3)
    rec = recipe.parse("copper 18; copper 17; copper 18")
    res = count_turns(img, params=CountParams(sections=3))
    lines, _, _ = recipe.evaluate(img, res.ring, res.angles, rec)
    assert any(not ln.ok and ln.section != "?" for ln in lines)
