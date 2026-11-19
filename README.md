# Toroid Turn Counter

Free Windows tool that counts the turns on a wound toroid from a digital microscope image
(live camera or a saved photo), so you can check the winding before case and potting.

## For users (no code)
1. Download `ToroidCounter.exe` and double-click it. Nothing to install, no license.
   (Windows SmartScreen may warn the first time because the exe is not signed: "More info" → "Run anyway".)
2. Pick your camera in the **Camera** list (USB cameras by name, or *Network / phone camera (URL)* for a
   phone app such as IP Webcam `http://<phone-ip>:8080/video`, or an RTSP camera). Press **Start camera**,
   position the core, press **Capture** (Space bar). Or use **Open image**.
3. Every counted turn gets a green line: a wire without a line is a missed turn. An orange ring marks a
   doubtful mark (colour does not look like wire): check it. Left-click adds a turn, right-click removes
   one, Ctrl+Z undoes. If the blue ring is wrong press **Set ring** (R) and follow the steps shown on the
   image: 1) centre of the hole, 2) inner edge (ID), 3) outer edge (OD). With a fixed camera and jig,
   tick **Keep ring** and the next parts reuse it.
4. Choose toroid, wire, layers and (optionally) expected turns to get PASS / FAIL and better warnings.
5. Optional **Recipe**: what each section of the core should have. Examples:
   - single winding: `copper 24`
   - two halves: `A: copper 20 ; B: copper 20`
   - three thirds: `copper 18; copper 18; copper 18`
   - half single, half bifilar: `A: copper 10 ; B: red 13 + green 13`

   The app splits the winding at the empty gaps between sections (pink lines), colours each mark by wire
   colour and shows a table with PASS / FAIL per section and colour. A grey mark means "colour not in the
   recipe" (usually glare or a lead): right-click it if it is not a turn. Keep a visible gap between sections.
6. Optional **ruler**: print `ruler/ruler_5in.pdf` at 100 % (check the 100 mm line), or laser-engrave the SVG.
   Put it next to the core, level with the top of the winding (3D-print the riser STL at that height).
   With both markers in view the app shows wound OD, ID and the wire pitch at the inner edge in mm.
7. **Confirm & log**. Results go to `Documents\ToroidCounter\results.csv` with the image and the overlay (ring + lines).
   Each confirmed part is also saved in `Documents\ToroidCounter\dataset` (image + YOLO labels from your
   corrected marks), so the counter can be retrained on your own parts. Enter = Confirm & log.

Good images: whole toroid in view, camera square to the part, diffuse or ring light (no shiny spots),
at least ~6 pixels per wire at the inner edge. A multi-layer winding cannot be fully counted from the
top: the app warns you.

You can edit the toroid, wire, recipe and wire-colour lists by copying `presets.json` into `Documents\ToroidCounter\`.

## For developers
```
pip install -r requirements-dev.txt
python app.py                 # run the GUI
python -m pytest -q tests     # synthetic accuracy tests
python tools/make_ruler.py ruler --riser-mm 8   # ruler PDF/PNG/SVG + riser STL
python tools/evaluate.py imgs # score on real labelled images (count in the file name: ..._052t_...)
```
Build the exe on Windows: `packaging\build_exe.bat` → `dist\ToroidCounter.exe`.
On GitHub, `.github/workflows/build-windows.yml` builds it on every push and attaches it to `v*` tags.

How it works and the known pitfalls: see [SPEC.md](SPEC.md).

Optional YOLO mode: `pip install ultralytics`, place a model trained on one class ("turn") at
`models/turns.pt`, tick **YOLO** in the toolbar.

## License
AGPL-3.0 (see `LICENSE`). Free to use, share and change; if you distribute a modified version, or run it
as a service, publish its source under the same license. This matches Ultralytics YOLO, which is AGPL-3.0.
