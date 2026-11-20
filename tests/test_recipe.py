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


# ------------------------------------------------------------------ recipe builder (v0.5)
def test_shares_and_names():
    secs = recipe.parse("[Filter A] A (1/4): copper 6 ; B (3/4): red 9 + green 9")
    assert [s.share for s in secs] == [0.25, 0.75]
    assert recipe.strip_name("[Filter A] A: copper 6") == ("Filter A", "A: copper 6")
    assert recipe.parse("A (25%): copper 3 ; B: copper 4")[0].share == 0.25
    txt = recipe.to_text(secs, "Filter A")
    assert txt == "[Filter A] A (1/4): copper 6 ; B (3/4): red 9 + green 9"
    assert recipe.parse(txt) == secs
    assert recipe.parse_shares("1/4, 3/4") == [0.25, 0.75]
    assert recipe.parse_shares("1 2 1") == [0.25, 0.5, 0.25]
    assert recipe.parse_shares("40%, 60%") == pytest.approx([0.4, 0.6])
    for bad in ("", "0, 1", "a, b"):
        with pytest.raises(ValueError):
            recipe.parse_shares(bad)
    assert recipe.to_text(recipe.parse("copper 24")) == "A: copper 24"  # single: no share shown


def test_save_and_load(tmp_path):
    store = tmp_path / "my_recipes.json"
    assert recipe.load_saved(store) == {"recipes": [], "layouts": {}}
    recipe.save_recipe(store, "[Choke] A (1/3): copper 18 ; B (1/3): copper 18 ; C (1/3): copper 18")
    recipe.save_recipe(store, "[Choke] A (1/2): copper 20 ; B (1/2): copper 20")  # same name replaces
    recipe.save_layout(store, "1/4 + 3/4", [0.25, 0.75])
    d = recipe.load_saved(store)
    assert d["recipes"] == ["[Choke] A (1/2): copper 20 ; B (1/2): copper 20"]
    assert d["layouts"] == {"1/4 + 3/4": [0.25, 0.75]}
    with pytest.raises(ValueError):
        recipe.save_recipe(store, "[Bad] copper twelve")


@pytest.mark.parametrize("seed", range(3))
def test_uneven_layout(seed):
    """1/4 + 3/4 with the same colour: the share decides which section is which."""
    secs = [{"copper": 6}, {"copper": 18}]
    img, _ = make_sectioned(secs, seed=seed + 20, shares=[0.25, 0.75])
    rec = recipe.parse("A (1/4): copper 6 ; B (3/4): copper 18")
    res = count_turns(img, params=CountParams(sections=2))
    lines, _, _ = recipe.evaluate(img, res.ring, res.angles, rec)
    got = {(ln.section, ln.expected): ln.counted for ln in lines if ln.section != "?"}
    assert got == {("A", 6): 6, ("B", 18): 18}
