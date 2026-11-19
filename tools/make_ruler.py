"""Make the 5 in calibration ruler card: SVG (laser / vector print), PDF + PNG (print at 100 %),
and an STL riser so the card sits at the same height as the top of the toroid.

Usage: python tools/make_ruler.py [out_folder] [--riser-mm 8]
"""
import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toroid_counter import measure as M  # noqa: E402

IN = 25.4


def shapes():
    """Card content as primitives in mm: ('rect', x, y, w, h, fill) / ('line', x0, y0, x1, y1, w) / ('text', x, y, size, s)."""
    out = [("rect", 0, 0, M.CARD_W, M.CARD_H, "none")]  # cut / outline
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (x, y) in M.MARKER_TL.items():
        bits = cv2.aruco.generateImageMarker(d, mid, 6)  # 4x4 data + 1 border module each side
        cell = M.MARKER_MM / 6
        for r in range(6):
            for c in range(6):
                if bits[r, c] == 0:
                    out.append(("rect", x + c * cell, y + r * cell, cell, cell, "black"))
        out.append(("text", x + 4.5, y + M.MARKER_MM + 4.2, 2.2, f"ID {mid}"))
    x0 = M.RULER_X0
    # inch scale on top edge, 1/16 in
    for k in range(5 * 16 + 1):
        x = x0 + k * IN / 16
        ln = 8 if k % 16 == 0 else 6 if k % 8 == 0 else 4.5 if k % 4 == 0 else 3 if k % 2 == 0 else 2
        out.append(("line", x, 0, x, ln, 0.15))
        if k % 16 == 0:
            out.append(("text", x - 0.9 if k else x + 0.5, 11.5, 3.2, str(k // 16)))
    # mm scale on bottom edge
    for k in range(int(M.RULER_LEN) + 1):
        x = x0 + k
        ln = 7 if k % 10 == 0 else 5 if k % 5 == 0 else 3
        out.append(("line", x, M.CARD_H, x, M.CARD_H - ln, 0.15))
        if k % 10 == 0:
            out.append(("text", x - (0.9 if k < 10 else 1.8 if k < 100 else 2.6), M.CARD_H - 8.2, 2.8, str(k)))
    out.append(("text", x0 + 30, 17.6, 2.6, "TOROID COUNTER  5 in / 127 mm  -  print at 100 %"))
    out.append(("text", x0 + 30, 21.2, 2.2, "keep card surface level with the top of the winding"))
    out.append(("text", 128.0, 11.5, 2.2, "in"))
    out.append(("text", M.RULER_X0 + M.RULER_LEN + 0.8, M.CARD_H - 1.2, 2.2, "mm"))
    return out


def write_svg(path: Path, page_w=215.9, page_h=279.4, ox=20, oy=25):
    items = []
    for s in shapes():
        if s[0] == "rect":
            _, x, y, w, h, fill = s
            st = 'fill="black"' if fill == "black" else 'fill="none" stroke="red" stroke-width="0.1"'
            items.append(f'<rect x="{x + ox:.3f}" y="{y + oy:.3f}" width="{w:.3f}" height="{h:.3f}" {st}/>')
        elif s[0] == "line":
            _, a, b, c, d, w = s
            items.append(f'<line x1="{a + ox:.3f}" y1="{b + oy:.3f}" x2="{c + ox:.3f}" y2="{d + oy:.3f}" '
                         f'stroke="black" stroke-width="{w}"/>')
        else:
            _, x, y, size, t = s
            items.append(f'<text x="{x + ox:.3f}" y="{y + oy:.3f}" font-family="Arial" font-size="{size}">{t}</text>')
    check_y = oy + M.CARD_H + 25
    items.append(f'<line x1="{ox}" y1="{check_y}" x2="{ox + 100}" y2="{check_y}" stroke="black" stroke-width="0.3"/>')
    items.append(f'<text x="{ox}" y="{check_y + 6}" font-family="Arial" font-size="3.5">Print check: this line must '
                 f'measure exactly 100 mm. Red outline = cut line (laser: cut red, engrave black).</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}mm" height="{page_h}mm" '
           f'viewBox="0 0 {page_w} {page_h}">\n' + "\n".join(items) + "\n</svg>\n")
    path.write_text(svg, encoding="utf-8")


def render_card(px_per_mm: float) -> np.ndarray:
    """Card alone as a grayscale image (used by tests to fake a photo)."""
    k = px_per_mm
    img = np.full((int(M.CARD_H * k) + 1, int(M.CARD_W * k) + 1), 255, np.uint8)
    P = lambda x, y: (int(round(x * k)), int(round(y * k)))  # noqa: E731
    for s in shapes():
        if s[0] == "rect" and s[5] == "black":
            _, x, y, w, h, _ = s
            cv2.rectangle(img, P(x, y), (P(x + w, y + h)[0] - 1, P(x + w, y + h)[1] - 1), 0, -1)
        elif s[0] == "line":
            _, a, b, c, d, w = s
            cv2.line(img, P(a, b), P(c, d), 0, max(1, int(w * k)))
    return img


def write_raster(png: Path, pdf: Path, dpi=600, page=(215.9, 279.4), ox=20, oy=25):
    k = dpi / IN
    W, Hh = int(page[0] * k), int(page[1] * k)
    img = np.full((Hh, W), 255, np.uint8)
    P = lambda x, y: (int(round((x + ox) * k)), int(round((y + oy) * k)))  # noqa: E731
    for s in shapes():
        if s[0] == "rect":
            _, x, y, w, h, fill = s
            if fill == "black":
                cv2.rectangle(img, P(x, y), (P(x + w, y + h)[0] - 1, P(x + w, y + h)[1] - 1), 0, -1)
            else:
                cv2.rectangle(img, P(x, y), P(x + w, y + h), 120, 2)
        elif s[0] == "line":
            _, a, b, c, d, w = s
            cv2.line(img, P(a, b), P(c, d), 0, max(1, int(w * k)))
        else:
            _, x, y, size, t = s
            cv2.putText(img, t, P(x, y), cv2.FONT_HERSHEY_SIMPLEX, size * k / 22, 0, max(1, int(size * k / 12)),
                        cv2.LINE_AA)
    cy = oy + M.CARD_H + 25
    cv2.line(img, P(0, cy), P(100, cy), 0, max(1, int(0.3 * k)))
    cv2.putText(img, "Print check: this line must measure exactly 100 mm", P(0, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5 * k / 22, 0, max(1, int(3.5 * k / 12)), cv2.LINE_AA)
    Image.fromarray(img).save(png, dpi=(dpi, dpi))
    Image.fromarray(img).convert("L").save(pdf, resolution=dpi)


def write_stl(path: Path, riser_mm: float):
    """Solid block the card is glued on: card size, riser_mm tall. Use riser = winding height."""
    w, d, h = M.CARD_W, M.CARD_H, riser_mm
    v = [(0, 0, 0), (w, 0, 0), (w, d, 0), (0, d, 0), (0, 0, h), (w, 0, h), (w, d, h), (0, d, h)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    with open(path, "wb") as f:
        f.write(b"toroid counter ruler riser".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            p = [np.array(v[i], float) for i in (a, b, c)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            n = n / np.linalg.norm(n)
            f.write(struct.pack("<12fH", *n, *p[0], *p[1], *p[2], 0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="ruler")
    ap.add_argument("--riser-mm", type=float, default=8.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_svg(out / "ruler_5in.svg")
    write_raster(out / "ruler_5in.png", out / "ruler_5in.pdf")
    write_stl(out / f"ruler_riser_{a.riser_mm:g}mm.stl", a.riser_mm)
    print("written to", out.resolve())
