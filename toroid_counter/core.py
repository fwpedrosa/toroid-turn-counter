"""Turn-counting core for wound toroids (top view, microscope image).

Pipeline
--------
1. Find the ring: centre + inner radius + outer radius (auto, or given by the operator).
2. Polar-unwrap the ring face so every turn becomes a vertical stripe.
3. Build an angular intensity profile in several radial bands of the top face.
4. Estimate the winding period with an FFT, then pick wire peaks with that spacing.
5. Cross-check bands and FFT; raise flags (gaps, low resolution, multi-layer, out of view).

The operator always confirms the result on the overlay and can add/remove marks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

N_ANGLES = 4096  # angular samples of the polar unwrap


@dataclass
class Ring:
    """Toroid position. A tilted camera sees the ring as an oval (ellipse):
    ratio = minor/major axis (1 = circle), phi = direction of the minor axis (radians).
    Radii and angles live in the "straightened" frame where the oval is a circle."""
    cx: float
    cy: float
    r_in: float
    r_out: float
    ratio: float = 1.0
    phi: float = 0.0

    def to_dict(self):
        return {"cx": self.cx, "cy": self.cy, "r_in": self.r_in, "r_out": self.r_out,
                "ratio": self.ratio, "phi": self.phi}

    def A(self) -> np.ndarray:
        """2x2 matrix: image offset from centre -> straightened offset."""
        c, s = math.cos(self.phi), math.sin(self.phi)
        R = np.array([[c, -s], [s, c]])
        return R @ np.diag([1.0 / self.ratio, 1.0]) @ R.T

    def pt(self, angle: float, r: float) -> tuple[float, float]:
        """Image point at straightened polar coordinates (angle, r)."""
        v = np.linalg.solve(self.A(), [r * math.cos(angle), r * math.sin(angle)])
        return float(self.cx + v[0]), float(self.cy + v[1])

    def polar_of(self, x: float, y: float) -> tuple[float, float]:
        """(angle, r) of an image point in the straightened frame."""
        v = self.A() @ [x - self.cx, y - self.cy]
        return float(math.atan2(v[1], v[0]) % (2 * math.pi)), float(math.hypot(*v))

    def angle_of(self, x: float, y: float) -> float:
        return self.polar_of(x, y)[0]


@dataclass
class CountParams:
    min_turns: int = 3
    max_turns: int = 600
    # radial bands sampled on the top face, as a fraction of (r_out - r_in) from r_in
    bands: tuple = ((0.15, 0.35), (0.35, 0.55), (0.55, 0.75))
    # optional physical data used only for warnings
    toroid_id_mm: float | None = None
    wire_od_mm: float | None = None
    expected_layers: int = 1
    sections: int = 1  # winding sections from the recipe; each one leaves an expected gap
    multi_color: bool = False  # recipe has several wire colours: skip the single-colour check


@dataclass
class CountResult:
    count: int
    angles: list  # radians, one per detected turn (best band)
    ring: Ring
    confidence: float  # 0..1
    fft_count: int
    band_counts: list
    flags: list = field(default_factory=list)
    profile: np.ndarray | None = None
    doubtful: list = field(default_factory=list)  # angles of marks whose colour is not wire-like
    gap_added: list = field(default_factory=list)  # angles added by the gap check

    def summary(self) -> str:
        s = f"{self.count} turns (confidence {self.confidence:.0%})"
        if self.flags:
            s += " | " + "; ".join(self.flags)
        return s


# --------------------------------------------------------------------------- ring
def preprocess(img: np.ndarray) -> np.ndarray:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clip specular glare so a few saturated highlights do not dominate the profile
    hi = np.percentile(gray, 99.5)
    gray = np.clip(gray, 0, hi).astype(np.float32)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def find_ring(img: np.ndarray) -> Ring | None:
    """Locate the toroid: largest blob that has a central hole.

    Tries both threshold polarities (toroid brighter or darker than background).
    Returns None when nothing ring-like is found (operator then sets it by clicks).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 800.0 / max(gray.shape))
    src = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if scale < 1:
        src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(src, (0, 0), 2), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = lab[..., 0]
    # background colour = median along the image border; toroid = anything far from it in
    # colour (red/green enamel, coloured core) or brightness
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((lab - bg) ** 2).sum(-1))
    diff = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # close the gaps between wires so the winding becomes one solid ring
    k = max(5, int(0.03 * max(g.shape)) | 1)
    best = None
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # low thresholds keep a bare core that is close to the background; high ones drop shadows.
    # Try several and keep the most ring-like result.
    for f in (0.4, 0.7, 1.0, 1.3):
        _, th = cv2.threshold(diff, f * otsu, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        contours, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is None:
            continue
        hier = hier[0]
        for i, c in enumerate(contours):
            if hier[i][3] != -1:  # want outer contours
                continue
            area = cv2.contourArea(c)
            if area < 0.02 * g.size:
                continue
            (ox, oy), r_out = cv2.minEnclosingCircle(c)
            # biggest hole inside this contour
            child = hier[i][2]
            hole, hole_area = None, 0
            while child != -1:
                a = cv2.contourArea(contours[child])
                if a > hole_area:
                    hole, hole_area = contours[child], a
                child = hier[child][0]
            if hole is None or hole_area < 0.06 * area:
                # real cores: the hole often looks dark (shadowed inner wall), not like background
                dark = _dark_hole(g, c)
                if dark is not None:
                    hole, hole_area = dark, cv2.contourArea(dark)
            if hole is None or hole_area < 0.01 * area:
                continue
            m = cv2.moments(hole)
            if m["m00"] == 0:
                continue
            hx, hy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            ring = _fit_oval(c, hole, hx, hy)
            # plausibility: a toroid is a compact blob mostly inside the picture. Big dark
            # clothing, a hand or the table merge with the frame edge and give a huge "ring".
            H, W = g.shape[:2]
            pts = c[:, 0, :]
            on_border = np.mean((pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= W - 2) | (pts[:, 1] >= H - 2))
            circ = np.array([ring.pt(a, ring.r_out) for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)])
            outside = np.mean((circ[:, 0] < 0) | (circ[:, 1] < 0) | (circ[:, 0] >= W) | (circ[:, 1] >= H))
            if outside > 0.25 or on_border > 0.2 or ring.r_out > 0.6 * min(H, W):
                continue
            off = math.hypot(ring.cx - ox, ring.cy - oy) / max(ring.r_out, 1)
            hole_frac = (ring.r_in / max(ring.r_out, 1)) ** 2
            # ring-like: hole centred, hole a sensible share of the ring (real cores ~0.15-0.5)
            score = (area * max(0.05, 1 - 3 * off) * (1.0 if 0.06 < hole_frac < 0.6 else 0.2)
                     * (1 - outside) ** 2)
            if best is None or score > best[0]:
                best = (score, Ring(ring.cx / scale, ring.cy / scale, ring.r_in / scale, ring.r_out / scale,
                                    ring.ratio, ring.phi))
    if best is None:
        return _hough_ring(gray)
    return best[1]


def _dark_hole(L: np.ndarray, outer) -> np.ndarray | None:
    """Darkest blob near the middle of a toroid-shaped region: the shadowed hole."""
    mask = np.zeros(L.shape, np.uint8)
    cv2.drawContours(mask, [outer], -1, 255, -1)
    inside = L[mask > 0]
    if inside.size < 100:
        return None
    m = cv2.moments(outer)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    (_, _), r_enc = cv2.minEnclosingCircle(outer)
    best = None
    for pct in (15, 25, 35):
        dark = ((L < np.percentile(inside, pct)) & (mask > 0)).astype(np.uint8) * 255
        k = max(3, int(0.04 * r_enc) | 1)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            a = cv2.contourArea(c)
            if a < 0.03 * inside.size or len(c) < 5:
                continue
            mm = cv2.moments(c)
            d = math.hypot(mm["m10"] / mm["m00"] - cx, mm["m01"] / mm["m00"] - cy) / r_enc
            fill = a / (math.pi * cv2.minEnclosingCircle(c)[1] ** 2 + 1e-9)  # roundness
            if d > 0.35 or fill < 0.45:
                continue
            score = fill * (1 - d)
            if best is None or score > best[0]:
                best = (score, c)
    return best[1] if best else None


def _fit_oval(outer, hole, hx, hy, min_tilt_ratio: float = 0.97) -> Ring:
    """Circle or ellipse from the outer contour; a lead sticking out is dropped before the final fit."""
    # the convex hull follows the wire tips, not the dips between sparse turns
    pts = cv2.convexHull(outer)[:, 0, :].astype(np.float32)
    ring = Ring(hx, hy, 1, 1)
    if len(pts) >= 12:
        for it in range(4):
            (ex, ey), (w, h), ang = cv2.fitEllipse(pts)
            ratio = min(w, h) / max(w, h)
            phi = math.radians(ang if w < h else ang + 90)
            trial = Ring(ex, ey, 1, 1, ratio, phi)
            d = np.array([trial.polar_of(x, y)[1] for x, y in pts])
            keep = np.abs(d / np.median(d) - 1) < (0.08, 0.06, 0.04, 0.04)[it]
            if keep.sum() < 12:
                break
            pts = pts[keep]
        hole_ratio, hole_phi = 1.0, phi
        if len(hole) >= 5:
            (_, _), (hw, hh), hang = cv2.fitEllipse(hole[:, 0, :].astype(np.float32))
            hole_ratio = min(hw, hh) / max(hw, hh)
            hole_phi = math.radians(hang if hw < hh else hang + 90)
        dphi = abs(math.sin(phi - hole_phi))
        # a real tilt squashes the hole the same way and keeps the centres together;
        # a bumpy outline (sparse turns, glare, shading on a light background) does not
        real_tilt = (ratio < min_tilt_ratio and hole_ratio < min_tilt_ratio + 0.01
                     and abs(ratio - hole_ratio) < 0.15 and dphi < 0.35
                     and math.hypot(ex - hx, ey - hy) < 0.15 * max(w, h) / 2)
        if real_tilt:
            # silhouette includes the side wall, the visible hole is shifted the other way: average
            ring = Ring((ex + hx) / 2, (ey + hy) / 2, 1, 1, ratio, phi)
    # median radius ignores a lead or a glare bump on a few contour points
    ring.r_in = float(np.median([ring.polar_of(x, y)[1] for x, y in hole[:, 0, :]]))
    ring.r_out = float(np.median([ring.polar_of(x, y)[1] for x, y in outer[:, 0, :]]))
    return ring


def straighten(gray: np.ndarray, ring: Ring) -> tuple[np.ndarray, Ring]:
    """Crop around the ring and stretch an oval back into a circle."""
    S = int(2 * ring.r_out * 1.12) + 1
    A = ring.A()
    t = np.array([S / 2, S / 2]) - A @ [ring.cx, ring.cy]
    M = np.hstack([A, t[:, None]]).astype(np.float64)
    out = cv2.warpAffine(gray, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, Ring(S / 2, S / 2, ring.r_in, ring.r_out)


def _hough_ring(gray: np.ndarray) -> Ring | None:
    g = cv2.medianBlur(gray, 5)
    h, w = g.shape
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=2, minDist=min(h, w) / 10,
                               param1=120, param2=60, minRadius=int(min(h, w) * 0.05),
                               maxRadius=int(max(h, w) * 0.6))
    if circles is None or len(circles[0]) < 2:
        return None
    c = circles[0]
    # two most concentric circles
    best = None
    for i in range(len(c)):
        for j in range(len(c)):
            if c[i][2] >= c[j][2]:
                continue
            d = math.hypot(c[i][0] - c[j][0], c[i][1] - c[j][1])
            if d < 0.1 * c[j][2] and (best is None or d < best[0]):
                best = (d, Ring(float(c[i][0]), float(c[i][1]), float(c[i][2]), float(c[j][2])))
    return best[1] if best else None


def ring_from_points(center, inner_pt, outer_pt, like: Ring | None = None) -> Ring:
    """Manual ring from 3 clicks. Keeps the oval (tilt) of a previous automatic ring if given."""
    cx, cy = center
    ring = Ring(cx, cy, 1, 1, like.ratio if like else 1.0, like.phi if like else 0.0)
    ring.r_in = ring.polar_of(*inner_pt)[1]
    ring.r_out = ring.polar_of(*outer_pt)[1]
    return ring


# --------------------------------------------------------------------------- unwrap
def unwrap(gray: np.ndarray, ring: Ring, n_angles: int = N_ANGLES) -> np.ndarray:
    """Polar image: rows = angle (0..2pi, n_angles rows), cols = radius 0..r_out*1.05."""
    max_r = ring.r_out * 1.05
    n_r = int(max(64, max_r))
    return cv2.warpPolar(gray, (n_r, n_angles), (ring.cx, ring.cy), max_r,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)


def _band_profile(polar: np.ndarray, ring: Ring, band) -> np.ndarray:
    max_r = ring.r_out * 1.05
    n_r = polar.shape[1]
    w = ring.r_out - ring.r_in
    c0 = int((ring.r_in + band[0] * w) / max_r * n_r)
    c1 = max(c0 + 1, int((ring.r_in + band[1] * w) / max_r * n_r))
    strip = polar[:, c0:c1].astype(np.float32)
    # median across radius is robust to a lead or a glare spot crossing the band
    prof = np.median(strip, axis=1)
    # remove slow illumination changes around the ring
    trend = gaussian_filter1d(prof, sigma=len(prof) / 60, mode="wrap")
    return prof - trend


def fft_period_count(profile: np.ndarray, kmin: int, kmax: int) -> tuple[int, float]:
    """Dominant number of cycles around the ring, and how much it stands out (0..1)."""
    spec = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    kmax = min(kmax, len(spec) - 1)
    if kmax <= kmin:
        return 0, 0.0
    seg = spec[kmin:kmax + 1]
    k = int(np.argmax(seg)) + kmin
    # prominence vs. median power in range; harmonics 2k are common, fundamental wins
    # if its sub-harmonic k/2 is also strong (alternating highlight) prefer the larger k anyway
    strength = float(seg.max() / (np.median(seg) * 50 + 1e-9))
    return k, min(1.0, strength)


def _peaks_circular(profile: np.ndarray, period: float) -> np.ndarray:
    n = len(profile)
    sm = gaussian_filter1d(profile, sigma=max(1.0, period / 8), mode="wrap")
    tri = np.concatenate([sm, sm, sm])
    prom = 0.15 * np.std(sm)
    pk, props = find_peaks(tri, distance=max(1, int(0.55 * period)), prominence=prom)
    keep = (pk >= n) & (pk < 2 * n)
    pk, pr = pk[keep] - n, props["prominences"][keep]
    if len(pr):
        # drop weak bumps (bare core, noise inside a gap) far below a typical wire
        pk = pk[pr > 0.3 * np.median(pr)]
    return np.sort(pk)


def _peak_colors(lab_r, ring_r, peaks, n):
    """Median Lab colour along the face of the wire at each peak (straightened frame)."""
    h, w = lab_r.shape[:2]
    radii = ring_r.r_in + np.linspace(0.2, 0.7, 7) * (ring_r.r_out - ring_r.r_in)
    out = []
    for p in peaks:
        a = p / n * 2 * np.pi
        xs = np.clip((ring_r.cx + radii * np.cos(a)).astype(int), 0, w - 1)
        ys = np.clip((ring_r.cy + radii * np.sin(a)).astype(int), 0, h - 1)
        out.append(np.median(lab_r[ys, xs].astype(np.float32), axis=0))
    c = np.array(out, np.float32).reshape(-1, 3)
    c[:, 0] *= 0.35  # brightness varies with shading/glare; hue matters more
    return c


def _color_outliers(cols):
    """(outlier mask, reference colour, distance threshold) for wire colours."""
    ref = np.median(cols, axis=0)
    d = np.sqrt(((cols - ref) ** 2).sum(1))
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    thr = max(14.0, float(np.median(d)) + 4.5 * float(mad))
    return d > thr, ref, thr


def _doubtful(lab_r, ring_r, peaks, n, ring=None, shape=None):
    """Marks whose colour does not match the wire (bare core, background, skin, a lead in
    shadow). They stay counted but are highlighted so the operator checks exactly those."""
    cols = _peak_colors(lab_r, ring_r, peaks, n)
    bad, _, _ = _color_outliers(cols)
    if ring is not None and shape is not None:  # outside the picture the colour means nothing
        H, W = shape[:2]
        for i, p in enumerate(peaks):
            x, y = ring.pt(p / n * 2 * np.pi, ring.r_in + 0.7 * (ring.r_out - ring.r_in))
            if not (0 <= x < W and 0 <= y < H):
                bad[i] = False
    if bad.sum() > 0.4 * len(peaks):  # colour too mixed to judge
        return np.zeros(len(peaks), bool)
    return bad


def _gap_fill(prof, peaks, n, lab_r, ring_r, check_color=True):
    """A gap about twice the normal pitch usually hides a turn whose peak was weak (lead on top,
    shadow, two turns touching). Add the best local maximum near the middle of such a gap."""
    peaks = np.sort(peaks)
    gaps = np.diff(np.concatenate([peaks, [peaks[0] + n]]))
    med = float(np.median(gaps))
    sm = gaussian_filter1d(prof, sigma=max(1.0, med / 8), mode="wrap")
    if check_color:
        _, ref, thr = _color_outliers(_peak_colors(lab_r, ring_r, peaks, n))
    new = []
    for i, g in enumerate(gaps):
        if not 1.65 * med <= g <= 2.6 * med:
            continue
        lo, hi = peaks[i] + 0.3 * g, peaks[i] + 0.7 * g
        idx = np.arange(int(lo), int(hi) + 1)
        vals = sm[idx % n]
        j = int(np.argmax(vals))
        if j in (0, len(idx) - 1):  # no local maximum inside: probably a real empty gap
            continue
        cand = idx[j] % n
        if check_color:
            c = _peak_colors(lab_r, ring_r, np.array([cand]), n)[0]
            if np.sqrt(((c - ref) ** 2).sum()) > thr:
                continue
        new.append(cand)
    return np.sort(np.concatenate([peaks, np.array(new, dtype=peaks.dtype)])) if new else peaks


def count_turns(img: np.ndarray, ring: Ring | None = None, params: CountParams | None = None) -> CountResult:
    params = params or CountParams()
    gray = preprocess(img)
    if ring is None:
        ring = find_ring(img)
        if ring is None:
            raise ValueError("Could not find the toroid. Click the centre, inner edge and outer edge.")
    flags: list[str] = []

    h, w = gray.shape
    edge = [ring.pt(a, ring.r_out) for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
    if any(x < 0 or y < 0 or x > w or y > h for x, y in edge):
        flags.append("TOROID EXCEEDS FIELD OF VIEW - count covers only the visible part")
    if ring.ratio < 0.97:
        tilt = math.degrees(math.acos(ring.ratio))
        flags.append(f"camera tilted ~{tilt:.0f} deg - corrected as an oval" +
                     ("; too steep, hidden turns likely" if tilt > 35 else ""))

    gray_r, ring_r = straighten(gray, ring)
    polar = unwrap(gray_r, ring_r)
    n = polar.shape[0]
    # physical limit: turns can't exceed inner circumference / smallest plausible wire (2 px)
    kmax = min(params.max_turns, int(2 * math.pi * ring.r_in / 2))

    band_results = []
    for band in params.bands:
        prof = _band_profile(polar, ring, band)
        k, strength = fft_period_count(prof, params.min_turns, kmax)
        if k == 0:
            continue
        peaks = _peaks_circular(prof, n / k)
        band_results.append((band, prof, k, strength, peaks))
    if not band_results:
        raise ValueError("Toroid too small in the image to count turns.")

    counts = [len(b[4]) for b in band_results]
    fft_counts = [b[2] for b in band_results]
    # pick the band whose peak count agrees best with its own FFT estimate
    best = min(band_results, key=lambda b: (abs(len(b[4]) - b[2]) / max(b[2], 1), -b[3]))
    _, prof, k, strength, peaks = best
    added_n = 0
    doubtful, gap_added = [], []
    if img.ndim == 3 and len(peaks) >= 4:
        lab_r, _ = straighten(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), ring)
        if params.sections <= 1:
            n_before, before = len(peaks), peaks.copy()
            peaks = _gap_fill(prof, peaks, n, lab_r, ring_r, check_color=not params.multi_color)
            added_n = len(peaks) - n_before
            gap_added = (np.setdiff1d(peaks, before) / n * 2 * math.pi).tolist()
        if not params.multi_color:
            bad = _doubtful(lab_r, ring_r, peaks, n, ring, img.shape)
            doubtful = (peaks[bad] / n * 2 * math.pi).tolist()
    count = len(peaks)
    angles = (peaks / n * 2 * math.pi).tolist()

    # ---- confidence & flags
    agree_fft = 1.0 - min(1.0, abs(count - k) / max(k, 1) * 5)
    spread = (max(counts) - min(counts)) / max(np.median(counts), 1)
    agree_bands = 1.0 - min(1.0, spread * 5)
    if params.sections > 1:  # sections break the single periodicity, so the FFT check says little
        confidence = max(0.0, min(1.0, 0.8 * agree_bands + 0.2))
    else:
        confidence = max(0.0, min(1.0, 0.45 * agree_fft + 0.35 * agree_bands + 0.2 * strength))

    if count > 1:
        gaps = np.diff(np.concatenate([peaks, [peaks[0] + n]]))
        med = np.median(gaps)
        big = np.where(gaps > 2.2 * med)[0]
        # with several sections the pitch differs per section; the recipe check covers gaps
        if len(big) and params.sections <= 1:
            flags.append(f"{len(big)} gap(s) / uneven spacing - check leads, crossovers or missing turns")
        pitch_px = 2 * math.pi * ring.r_in / count  # tightest spacing, at the inner edge
        if pitch_px < 6:
            confidence *= 0.5
            flags.append(f"LOW RESOLUTION: only {pitch_px:.1f} px per wire - zoom in or use a higher resolution")
    if params.sections <= 1 and abs(count - k) / max(k, 1) > 0.05:
        flags.append(f"peak count and periodicity disagree ({count} vs {k}) - verify on overlay")
    if spread > 0.05:
        flags.append(f"radial bands disagree {counts} - verify on overlay")

    # multi-layer check needs physical data
    if params.toroid_id_mm and params.wire_od_mm:
        max_single = int(math.pi * (params.toroid_id_mm - params.wire_od_mm) / params.wire_od_mm)
        if params.expected_layers > 1:
            flags.append(f"MULTI-LAYER ({params.expected_layers}): camera sees only the top layer")
        elif count > max_single:
            flags.append(f"count {count} > max single layer {max_single} - wrong preset or multi-layer")
        elif count > 0.9 * max_single:
            flags.append("winding is nearly full - check for a second layer")
    elif params.expected_layers > 1:
        flags.append(f"MULTI-LAYER ({params.expected_layers}): camera sees only the top layer")

    if added_n:
        flags.append(f"{added_n} weak turn(s) added in a double-width gap - check the marks with a white ring")
    if doubtful:
        flags.append(f"{len(doubtful)} doubtful mark(s) (orange, colour is not like the wire) - check them")
        confidence *= 0.85
    res = CountResult(count=count, angles=angles, ring=ring, confidence=confidence,
                      fft_count=k, band_counts=counts, flags=flags, profile=prof)
    res.doubtful = doubtful
    res.gap_added = gap_added
    return res


# --------------------------------------------------------------------------- overlay
def draw_overlay(img: np.ndarray, ring: Ring, angles, removed=(), added=(), label_every: int = 0,
                 dot_colors: dict | None = None, borders=(), lines: bool = True,
                 doubtful=(), ring_label: bool = True) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(1, int(round(max(out.shape[:2]) / 900)))
    c = (int(ring.cx), int(ring.cy))

    def P(a, r):
        x, y = ring.pt(a, r)
        return int(round(x)), int(round(y))

    circ = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    for r in (ring.r_in, ring.r_out):
        cv2.polylines(out, [np.array([P(a, r) for a in circ], np.int32)], True, (255, 200, 0), t, cv2.LINE_AA)
    cv2.drawMarker(out, c, (255, 200, 0), cv2.MARKER_CROSS, 20 * t, t)
    r_mark = ring.r_in + 0.45 * (ring.r_out - ring.r_in)
    all_angles = sorted(list(angles) + list(added))
    if not label_every:
        label_every = 1 if len(all_angles) <= 60 else (5 if len(all_angles) <= 300 else 10)
    dot = max(2, int(0.25 * 2 * math.pi * r_mark / max(len(all_angles), 1)))
    dot = min(dot, 6 * t)
    for i, a in enumerate(all_angles):
        col = (0, 255, 255) if a in added else (0, 220, 0)
        if dot_colors and a in dot_colors:
            col = dot_colors[a]
        if lines:  # a line along each counted turn: a wire without a line = missed turn
            cv2.line(out, P(a, ring.r_in + 0.05 * (ring.r_out - ring.r_in)),
                     P(a, ring.r_out * 1.02), (0, 0, 0), t + 2, cv2.LINE_AA)
            cv2.line(out, P(a, ring.r_in + 0.05 * (ring.r_out - ring.r_in)),
                     P(a, ring.r_out * 1.02), col, t, cv2.LINE_AA)
        cv2.circle(out, P(a, r_mark), dot, col, -1)
        if a in added:  # manual marks get a white ring so they stay visible in colour mode
            cv2.circle(out, P(a, r_mark), dot + t, (255, 255, 255), t)
        if (i + 1) % label_every == 0 or i == 0:
            lx, ly = P(a, ring.r_out + 14 * t)
            cv2.putText(out, str(i + 1), (lx - 8 * t, ly + 4 * t), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * t, (0, 0, 0), max(1, t // 2 + 1) + 2, cv2.LINE_AA)
            cv2.putText(out, str(i + 1), (lx - 8 * t, ly + 4 * t), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * t, (0, 255, 0), max(1, t // 2 + 1), cv2.LINE_AA)
    for a in doubtful:  # orange ring = check this one
        if any(abs(a - b) < 1e-9 for b in all_angles):
            cv2.circle(out, P(a, r_mark), dot + 3 * t, (0, 140, 255), 2 * t, cv2.LINE_AA)
    if ring_label:
        cv2.putText(out, "ring", P(-math.pi / 2, ring.r_out * 1.06), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * t, (255, 200, 0), max(1, t), cv2.LINE_AA)
    for a in borders:
        cv2.line(out, P(a, 0.9 * ring.r_in), P(a, 1.1 * ring.r_out), (255, 0, 255), 2 * t, cv2.LINE_AA)
    for a in removed:
        cv2.drawMarker(out, P(a, r_mark), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 4 * dot, t)
    return out
