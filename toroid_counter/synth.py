"""Synthetic top-view images of wound toroids with a known number of turns.

Used for automated tests and to tune the counter before real images exist.
Real photos (see SPEC.md) are still required before trusting the tool.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def make_toroid(turns: int, size: int = 1200, od_frac: float = 0.8, id_frac: float = 0.45,
                wire_fill: float = 0.85, jitter: float = 0.08, gap_deg: float = 0.0,
                noise: float = 6.0, glare: bool = True, lead: bool = True, blur: float = 1.0,
                rotation: float | None = None, offset=(0, 0), dark_bg: bool = True,
                seed: int = 0) -> np.ndarray:
    """Return a BGR uint8 image.

    wire_fill : wire diameter as a fraction of the pitch at the inner edge (1 = touching).
    gap_deg   : an empty sector (partial winding), in degrees.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    cx, cy = w / 2 + offset[0], h / 2 + offset[1]
    r_out = od_frac * size / 2
    r_in = id_frac * r_out
    rot = rng.uniform(0, 2 * np.pi) if rotation is None else rotation

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    th = np.mod(np.arctan2(dy, dx) - rot, 2 * np.pi)

    bg = 25.0 if dark_bg else 215.0
    img = np.full((h, w), bg, np.float32)
    ring = (r >= r_in * 0.97) & (r <= r_out * 1.03)
    img[ring] = 70.0  # core (mostly hidden by wire)

    span = 2 * np.pi - math.radians(gap_deg)
    pitch = span / turns
    base = np.arange(turns) * pitch + rng.normal(0, jitter * pitch, turns)
    # wire radius in px, set by the inner edge where turns are tightest
    wire_r = wire_fill * (2 * math.pi * r_in / (2 * math.pi / pitch)) / 2
    idx = np.round(th / pitch).astype(int)
    wire_val = np.zeros_like(img)
    hit = np.zeros(img.shape, bool)
    for d in (-1, 0, 1):
        j = np.mod(idx + d, turns)
        dth = np.angle(np.exp(1j * (th - base[j])))
        s = np.abs(dth) * np.maximum(r, 1)  # arc distance to wire axis in px
        inside = (s < wire_r) & ring
        u = np.clip(s / wire_r, 0, 1)
        shade = np.sqrt(1 - u ** 2)
        val = 60 + 140 * shade + 55 * np.exp(-((u - 0.2) ** 2) / 0.02)  # copper + specular
        better = inside & (val > wire_val)
        wire_val[better] = val[better]
        hit |= inside
    img[hit] = wire_val[hit]

    if lead:
        ang = rot + rng.uniform(0, 2 * np.pi)
        p0 = (int(cx + r_out * math.cos(ang)), int(cy + r_out * math.sin(ang)))
        p1 = (int(cx + 1.6 * r_out * math.cos(ang + 0.3)), int(cy + 1.6 * r_out * math.sin(ang + 0.3)))
        cv2.line(img, p0, p1, 180, max(2, int(wire_r * 2)), cv2.LINE_AA)

    if glare:
        gx, gy = cx + rng.uniform(-0.6, 0.6) * r_out, cy + rng.uniform(-0.6, 0.6) * r_out
        gs = 0.08 * r_out
        img += 90 * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * gs ** 2))

    # uneven lighting
    img *= 0.85 + 0.15 * (xx / w)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    f = img.astype(np.float32)
    col = np.dstack([f * 0.55, f * 0.75, f]).astype(np.uint8)  # copper tint
    return col


COLORS = {  # BGR of enamel under white light
    "copper": (60, 120, 200),
    "red": (40, 40, 210),
    "green": (60, 170, 60),
    "blue": (200, 110, 40),
    "yellow": (40, 200, 220),
}


def make_sectioned(sections, size: int = 1400, od_frac: float = 0.8, id_frac: float = 0.45,
                   wire_fill: float = 0.85, sector_gap_deg: float = 25.0, noise: float = 5.0,
                   jitter: float = 0.05, glare: bool = True, seed: int = 0, shares=None):
    """Toroid with several winding sections, e.g. [{"copper": 10}, {"red": 13, "green": 13}].

    Each section covers an equal share of the ring (or `shares`, fractions adding up to 1); a multi-colour section is wound
    multifilar (colours alternate side by side). Returns (BGR image, list of (angle, colour)).
    """
    rng = np.random.default_rng(seed)
    h = w = size
    cx = cy = size / 2
    r_out = od_frac * size / 2
    r_in = id_frac * r_out
    rot = rng.uniform(0, 2 * np.pi)
    shares = list(shares) if shares else [1 / len(sections)] * len(sections)
    starts = np.concatenate([[0.0], np.cumsum(shares)[:-1]]) * 2 * np.pi
    gap = math.radians(sector_gap_deg)
    wires = []
    for si, sec in enumerate(sections):
        share = shares[si] * 2 * np.pi
        cols = list(sec.keys())
        n = max(sec.values())
        total = sum(sec.values())
        pitch = (share - gap) / total
        start = starts[si] + gap / 2
        k = 0
        for t in range(n):
            for c in cols:
                if t < sec[c]:
                    wires.append(((start + (k + 0.5) * pitch + rng.normal(0, jitter * pitch)) % (2 * np.pi), c))
                    k += 1
    wires.sort()
    ang = np.array([a for a, _ in wires])
    min_pitch = np.min(np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]])))
    wire_r = wire_fill * r_in * min_pitch / 2

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    th = np.mod(np.arctan2(dy, dx) - rot, 2 * np.pi)
    ring = (r >= r_in * 0.97) & (r <= r_out * 1.03)
    img = np.full((h, w, 3), 25.0, np.float32)
    img[ring] = (70, 75, 80)
    best_s = np.full((h, w), np.inf, np.float32)
    best_i = np.zeros((h, w), int)
    pos = np.searchsorted(ang, th)
    for d in (-1, 0):
        j = np.mod(pos + d, len(ang))
        s = np.abs(np.angle(np.exp(1j * (th - ang[j])))) * np.maximum(r, 1)
        better = s < best_s
        best_s[better] = s[better]
        best_i[better] = j[better]
    inside = (best_s < wire_r) & ring
    u = np.clip(best_s / wire_r, 0, 1)
    shade = np.sqrt(1 - u ** 2)
    spec = 70 * np.exp(-((u - 0.2) ** 2) / 0.02)
    palette = np.array([COLORS[c] for _, c in wires], np.float32)
    colimg = palette[best_i] * (0.35 + 0.75 * shade)[..., None] + spec[..., None]
    img[inside] = colimg[inside]
    if glare:
        gx, gy = cx + rng.uniform(-0.6, 0.6) * r_out, cy + rng.uniform(-0.6, 0.6) * r_out
        gs = 0.07 * r_out
        img += (70 * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * gs ** 2)))[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, noise, img.shape)
    truth = [((a + rot) % (2 * np.pi), c) for a, c in wires]
    return np.clip(img, 0, 255).astype(np.uint8), truth


def tilt(img: np.ndarray, deg: float, direction_deg: float = 30.0) -> np.ndarray:
    """Approximate a camera tilted by `deg`: squash the image by cos(deg) along one direction."""
    h, w = img.shape[:2]
    c, s = math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))
    R = np.array([[c, -s], [s, c]])
    A = R @ np.diag([math.cos(math.radians(deg)), 1.0]) @ R.T
    t = np.array([w / 2, h / 2]) - A @ [w / 2, h / 2]
    M = np.hstack([A, t[:, None]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
