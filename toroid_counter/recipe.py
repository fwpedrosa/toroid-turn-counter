"""Recipes: expected turns per winding section and per wire colour.

Text format (one line, sections split by ';', colours by '+'):
    A: copper 10 ; B: red 13 + green 13
A section without a name is fine:  "copper 24"  or  "red 8 + green 8".

The app splits the detected turns into sections at the largest empty gaps around the
ring, gives each turn a colour (nearest reference colour, refined by k-means on the
image itself), matches sections to the recipe and reports PASS / FAIL per line.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

import cv2
import numpy as np

# reference enamel colours (BGR) under white light; editable in presets.json "colors"
DEFAULT_COLORS = {
    "copper": (60, 120, 200),
    "red": (40, 40, 210),
    "green": (60, 170, 60),
    "blue": (200, 110, 40),
    "yellow": (40, 200, 220),
    "black": (30, 30, 30),
    "white": (220, 220, 220),
}
# colour used for the overlay dot of each class
DOT_BGR = {"copper": (0, 165, 255), "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 120, 0),
           "yellow": (0, 255, 255), "black": (80, 80, 80), "white": (255, 255, 255)}


@dataclass
class Section:
    name: str
    turns: dict  # colour -> expected turns


def parse(text: str) -> list[Section]:
    sections = []
    for i, part in enumerate(p for p in text.split(";") if p.strip()):
        name = chr(ord("A") + i)
        if ":" in part:
            name, part = part.split(":", 1)
            name = name.strip()
        turns = {}
        for item in part.split("+"):
            m = re.match(r"\s*([A-Za-z_-]+)\s*[x=]?\s*(\d+)\s*$", item)
            if not m:
                raise ValueError(f"Could not read '{item.strip()}'. Use e.g. 'red 13 + green 13'.")
            turns[m.group(1).lower()] = int(m.group(2))
        sections.append(Section(name, turns))
    if not sections:
        raise ValueError("Empty recipe.")
    return sections


def to_text(sections: list[Section]) -> str:
    return " ; ".join(f"{s.name}: " + " + ".join(f"{c} {n}" for c, n in s.turns.items()) for s in sections)


# --------------------------------------------------------------------------- colour
def sample_colors(img: np.ndarray, ring, angles) -> np.ndarray:
    """Median BGR along the face of each turn (radial line from 20 % to 70 % of the face)."""
    h, w = img.shape[:2]
    out = []
    radii = ring.r_in + np.linspace(0.2, 0.7, 9) * (ring.r_out - ring.r_in)
    for a in angles:
        pts = np.array([ring.pt(a, r) for r in radii])
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        out.append(np.median(img[ys, xs].reshape(-1, 3), axis=0))
    return np.array(out, np.float32).reshape(-1, 3)


def _lab(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    lab = lab.reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 0.35  # brightness varies with glare and shading: weigh hue/chroma more
    return lab


def classify(img, ring, angles, names: list[str], refs: dict | None = None) -> list[str]:
    """One colour name per angle, restricted to the colours named in the recipe."""
    if not len(angles) or not names:
        return [""] * len(angles)
    refs = {**DEFAULT_COLORS, **(refs or {})}
    x = _lab(sample_colors(img, ring, angles))
    centers = _lab(np.array([refs.get(n, (128, 128, 128)) for n in names], np.float32))
    if len(names) == 1:  # single colour: the image itself defines it, only outliers matter
        centers[0] = np.median(x, axis=0)
    for _ in range(5):  # k-means seeded with the references adapts to the real lighting
        d = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(len(names)):
            if (lab == k).any():
                centers[k] = 0.5 * centers[k] + 0.5 * x[lab == k].mean(0)
    dist = np.sqrt(d[np.arange(len(lab)), lab])
    # far from every colour in the recipe (glare spot, lead, core): mark as unknown
    odd = dist > max(25.0, 3.0 * float(np.median(dist)))
    return ["?" if o else names[k] for k, o in zip(lab, odd)]


# --------------------------------------------------------------------------- sections
def split_sections(angles, n: int) -> list[list[int]]:
    """Indices of the (sorted) angles in each section, cut at the n largest gaps."""
    m = len(angles)
    if n <= 1 or m < n:
        return [list(range(m))]
    a = np.asarray(angles)
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))  # gap after each turn
    cuts = sorted(np.argsort(gaps)[-n:])
    groups = []
    for i, c in enumerate(cuts):
        start = c + 1
        end = cuts[(i + 1) % n] + 1
        idx = [(start + k) % m for k in range(((end - start) % m) or m)]
        groups.append(idx)
    return groups


def gap_angles(angles, n: int) -> list[float]:
    """Angle in the middle of each of the n largest gaps (for drawing section borders)."""
    a = np.asarray(sorted(angles))
    if n <= 1 or len(a) < n:
        return []
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return [float((a[c] + gaps[c] / 2) % (2 * np.pi)) for c in np.argsort(gaps)[-n:]]


@dataclass
class Line:
    section: str
    color: str
    expected: int
    counted: int

    @property
    def ok(self):
        return self.expected == self.counted


def evaluate(img, ring, angles, recipe: list[Section], refs=None):
    """Return (lines, colour per sorted angle, sorted angles)."""
    angles = sorted(angles)
    names = sorted({c for s in recipe for c in s.turns})
    colors = classify(img, ring, angles, names, refs)
    groups = split_sections(angles, len(recipe))
    counts = [{c: sum(1 for i in g if colors[i] == c) for c in names} for g in groups]
    if len(groups) != len(recipe):  # not enough turns to split: one bucket for everything
        total = {c: sum(1 for x in colors if x == c) for c in names}
        lines = [Line("all", c, sum(s.turns.get(c, 0) for s in recipe), total[c]) for c in names]
        return _unknown(lines, colors), colors, angles
    # match detected sections to recipe sections (few sections: try every order)
    best = None
    for perm in itertools.permutations(range(len(recipe))):
        cost = sum(abs(counts[g].get(c, 0) - recipe[r].turns.get(c, 0))
                   for g, r in enumerate(perm) for c in names)
        if best is None or cost < best[0]:
            best = (cost, perm)
    lines = []
    for g, r in sorted(enumerate(best[1]), key=lambda t: t[1]):
        sec = recipe[r]
        for c in names:
            if c in sec.turns or counts[g].get(c, 0):
                lines.append(Line(sec.name, c, sec.turns.get(c, 0), counts[g].get(c, 0)))
    return _unknown(lines, colors), colors, angles


def _unknown(lines, colors):
    n = colors.count("?")
    if n:
        lines.append(Line("?", "unknown colour - check", 0, n))
    return lines
