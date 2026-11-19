# Toroid Turn Counter: spec and plan (v0.2)

## Goal
Count the turns on a wound toroid from a top-view digital microscope image (live camera or a saved
photo), for any toroid and wire size, **before** the part goes into its case and potting.
The operator always sees the result on the image and confirms it. Every confirmed result is logged.

Free tool, shipped as one Windows `.exe` (no Python, no code, no license key, no expiry).
Source is AGPL-3.0 because the optional YOLO path uses Ultralytics (AGPL-3.0).

## How it counts (baseline: OpenCV, no AI)
1. **Find the ring.** Background level is taken from the image border; everything that differs
   from it is "toroid". The biggest blob with a hole gives centre, inner radius and outer radius
   (median distances, so a lead wire does not stretch the ring). Hough circles as fallback.
   If it still fails, the operator clicks **centre, inner edge, outer edge** (3 clicks).
2. **Unwrap.** `cv2.warpPolar` turns the ring face into a strip where every turn is a stripe.
3. **Profile.** Median brightness across three radial bands of the top face
   (15-35 %, 35-55 %, 55-75 % of the face width), slow lighting trend removed.
4. **Period + peaks.** An FFT gives the dominant number of cycles around the ring (≈ turns).
   Peaks are then picked with that spacing; weak bumps (bare core, noise in a gap) are dropped.
5. **Cross-check.** Count from peaks vs. FFT, and the three bands vs. each other, give the
   confidence score. Any disagreement becomes a visible warning.
6. **Operator check.** Green dots + numbers on the overlay. Left-click adds a turn (yellow),
   right-click removes the nearest one (red X). Optional "Expected turns" shows PASS / FAIL.
7. **Recipe (optional).** Expected turns per section and per wire colour, any number of sections
   (`copper 24`, `copper 18; copper 18; copper 18`, `A: copper 10 ; B: red 13 + green 13`).
   Sections are cut at the N largest gaps; each mark gets a colour by sampling the wire along its
   face (Lab colour, k-means seeded with reference colours from presets.json); sections are matched
   to the recipe in the order that fits best. Marks whose colour is far from every recipe colour are
   grey "?" (glare, lead) and turn the result into CHECK instead of PASS.
8. **Log.** `Documents\ToroidCounter\results.csv` + the raw image and the overlay per result.

Optional **YOLO** mode (detect each wire crossing as a box) is for hard cases once a trained
model exists (`models/turns.pt`). It is off by default and not in the standard exe because PyTorch
adds ~1 GB. The rest of the app (overlay, correction, log) is shared.

## Known pitfalls and what the app does about them
| Pitfall | Effect | Handling |
|---|---|---|
| Multi-layer winding | Camera sees only the top layer, count is too low | "Layers" field warns; with toroid + wire presets the app warns when the count is above / near the single-layer maximum π·(ID−d)/d |
| Enamel glare / specular spots | Fake or merged peaks | Top 0.5 % brightness clipped, CLAHE, median across radius. **Use diffuse / ring light, polarizer if possible** |
| Toroid larger than field of view | Partial count | Flag "exceeds field of view". Fix: lower zoom. (Later: rotate-and-stitch mode) |
| Start / end leads, crossovers | Extra peak or local mess | Median across radius ignores a thin lead; gap/uneven-spacing flag; operator removes the extra mark |
| Gaps, uneven spacing, partial winding | FFT and peak count disagree | Both flagged; peaks still counted one by one |
| Thin wire vs. resolution | Neighbouring wires merge | "LOW RESOLUTION" flag below 6 px per wire at the inner edge, confidence halved. Rule: ≥ 6 px per wire at the inner edge |
| Wires touching at inner edge | No dark gap between turns | Counting uses the face, not the inner edge, where turns spread apart |
| Toroid off-centre / tilted | Ellipse, not circle | Centre comes from the hole; tilt > ~10° not supported yet: keep camera square to the part |
| Light vs. dark background | Wrong segmentation | Background measured from the border, works with both |
| Bifilar / multi-colour sections | Two wires side by side, half the pitch | Counted as separate turns, then split by colour per section |
| Copper vs. red enamel look alike | Wrong colour class | k-means adapts to the real light; add your own reference colours in presets.json; dataset should include each colour pair |

Rule for the operator: **a count is never accepted only because the software said so.** The
"Confirm & log" button asks again when there are warnings.

## Test-image dataset plan (real images, before trusting the tool)
Target **60-100 images with a known count**, collected with the real microscope and lighting.

- **Coverage matrix**: each toroid size you build (e.g. T50, T68, T80, T106, T130, T200) ×
  thin / medium / thick wire × low / medium / high turn count. At least 3 images per cell
  that you use in production.
- **Recipes you build**: for each recipe (single, halves, thirds, half + bifilar red/green) at least 5 images;
  put the recipe in a sidecar text file with the same name (`..._001.txt`).
- **Hard cases on purpose (≈ 25 %)**: glare, leads crossing the face, uneven spacing,
  partial winding with a gap, nearly full winding, multi-layer, part slightly off-centre,
  part bigger than the view, lower zoom.
- **Ground truth**: count by hand twice (or use the winding machine counter). Write the count in
  the file name: `T68_AWG24_052t_001.jpg` (= 52 turns).
- **Same part, several shots**: rotate the part 90°, re-shoot, re-light: tests repeatability.
- **Score**: `python tools/evaluate.py <folder>` prints exact-match rate, within ±1 rate and how
  many wrong results were flagged. Goal for v1: ≥ 95 % exact on normal parts and **every wrong
  result flagged** (a wrong count without a warning is the worst failure).
- Keep 20 % of the images aside, never used for tuning, for the final check.
- The same images (labelled with boxes) become the YOLO training set if OpenCV is not enough.

Until real images exist the counter is tested with synthetic toroids
(`tests/test_core.py`, 10-300 turns, light/dark background, glare, leads, gaps, jitter).

## Roadmap
1. v0.2 (this): OpenCV counter, GUI, manual correction, presets, recipes (sections + colours), CSV log, exe build.
2. Collect the dataset above, tune thresholds, publish v0.3.
3. If accuracy is not enough on some parts: label boxes, train YOLO, ship a "YOLO" exe build.
4. Later: rotate-and-stitch for large toroids, ellipse fit for tilt, per-part recipes.
