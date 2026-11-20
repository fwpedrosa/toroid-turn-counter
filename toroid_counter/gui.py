"""Tkinter GUI: pick a camera (or open an image), capture, count, correct, confirm, log, learn."""
from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import __version__, builder, cameras, core, logbook, measure, recipe, yolo_backend

HELP = (
    "QUICK START\n"
    "1. Pick your camera in the list (or a phone / network URL) and press Start camera.\n"
    "   Position the core, then press Capture (or the Space bar). Or use Open image.\n"
    "2. The app finds the ring and counts. Every counted turn gets a green line.\n"
    "   A wire without a line = missed turn. An orange ring = doubtful mark, check it.\n"
    "3. Fix mistakes: LEFT-click adds a turn, RIGHT-click removes the nearest one, Ctrl+Z undoes.\n"
    "4. If the blue ring is wrong, press Set ring (R) and follow the steps on the image:\n"
    "   1) centre of the hole   2) inner edge (ID)   3) outer edge of the winding (OD).\n"
    "   Tick 'Keep ring' when the camera and the core position are fixed: the next parts reuse it.\n"
    "5. Press Confirm & log (Enter). Image, overlay with the ring, and result go to\n"
    "   Documents\\ToroidCounter. Each confirmed part is also saved as a training sample\n"
    "   (dataset folder) so the counter can learn from your corrections.\n\n"
    "Recipe (optional): press Build... next to Recipe. Pick a layout (single, halves, thirds,\n"
    "quarters, or Custom with your own shares such as 1/4 + 3/4), type the turns per section,\n"
    "tick Bifilar for two wires, and Save recipe as... to pick it from the list next time.\n"
    "Tips: camera straight down, diffuse or ring light, whole toroid in view, plain background."
)

RING_STEPS = ["Step 1 of 3:  click the CENTRE of the hole",
              "Step 2 of 3:  click the INNER edge of the winding (ID)",
              "Step 3 of 3:  click the OUTER edge of the winding (OD)"]


def resource(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    p = base / name
    return p if p.exists() else Path(__file__).resolve().parent / name


def load_presets() -> dict:
    user = logbook.default_dir() / "presets.json"
    for p in (user, resource("presets.json")):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {"toroids": [{"name": "(none)", "id_mm": None}], "wires": [{"name": "(none)", "od_mm": None}]}


def recipes_path() -> Path:
    return logbook.default_dir() / "my_recipes.json"


def settings_path() -> Path:
    return logbook.default_dir() / "settings.json"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Toroid Turn Counter {__version__}")
        self.geometry("1320x840")
        self.minsize(980, 640)
        self.presets = load_presets()
        self.img = None            # BGR frame being analysed
        self.result = None
        self.meas = None
        self.marks: list[float] = []      # current turn angles (radians)
        self.added: list[float] = []
        self.removed: list[float] = []
        self.undo_stack: list[tuple[str, float]] = []
        self.ring = None
        self.ring_clicks: list = []
        self.mouse = None
        self.mode = "edit"         # "edit" or "ring"
        self.cap = None
        self.cam_was_on = False
        self.live_job = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.lines = None
        self._build()
        self._load_settings()
        self.refresh_cameras()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ layout
    def _build(self):
        style = ttk.Style(self)
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"))
        bar = ttk.Frame(self, padding=4)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open image", command=self.open_image).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(bar, text="Camera").pack(side=tk.LEFT)
        self.cam_choice = tk.StringVar()
        self.cam_box = ttk.Combobox(bar, textvariable=self.cam_choice, width=30, state="readonly")
        self.cam_box.pack(side=tk.LEFT, padx=2)
        self.cam_box.bind("<<ComboboxSelected>>", self.on_camera_choice)
        ttk.Button(bar, text="⟳", width=3, command=self.refresh_cameras).pack(side=tk.LEFT)
        self.cam_btn = ttk.Button(bar, text="Start camera", command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Capture [Space]", style="Big.TButton", command=self.capture).pack(side=tk.LEFT, padx=2)
        self.live_count = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Live count", variable=self.live_count).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="Set ring [R]", command=self.start_ring_mode).pack(side=tk.LEFT)
        self.keep_ring = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Keep ring", variable=self.keep_ring).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Count again", command=self.recount).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Undo [Ctrl+Z]", command=self.undo).pack(side=tk.LEFT)
        ttk.Button(bar, text="Reset marks", command=self.reset_marks).pack(side=tk.LEFT, padx=2)
        self.use_yolo = tk.BooleanVar(value=False)
        yb = ttk.Checkbutton(bar, text="YOLO", variable=self.use_yolo)
        yb.pack(side=tk.LEFT, padx=4)
        if not yolo_backend.available():
            yb.state(["disabled"])
        ttk.Button(bar, text="Help", command=lambda: messagebox.showinfo("How to use", HELP)).pack(side=tk.RIGHT)

        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_left)
        self.canvas.bind("<Button-3>", self.on_right)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        side = ttk.Frame(main, padding=8, width=340)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        self.count_lbl = tk.Label(side, text="--", font=("Segoe UI", 44, "bold"))
        self.count_lbl.pack(anchor="w")
        self.conf_lbl = ttk.Label(side, text="Pick a camera and press Start camera, or Open image.")
        self.conf_lbl.pack(anchor="w")
        self.meas_lbl = ttk.Label(side, text="", foreground="#004080", justify="left")
        self.meas_lbl.pack(anchor="w")
        self.pass_lbl = tk.Label(side, text="", font=("Segoe UI", 16, "bold"))
        self.pass_lbl.pack(anchor="w", pady=(4, 4))
        self.table = ttk.Treeview(side, columns=("sec", "col", "exp", "cnt", "res"), show="headings", height=4)
        for c, t, wd in (("sec", "Section", 60), ("col", "Colour", 80), ("exp", "Exp.", 45),
                         ("cnt", "Count", 50), ("res", "", 55)):
            self.table.heading(c, text=t)
            self.table.column(c, width=wd, anchor="center")
        self.table.tag_configure("bad", foreground="#c00000")
        self.table.tag_configure("good", foreground="#008000")
        self.table.pack(fill=tk.X, pady=(0, 4))
        self.flags_txt = tk.Text(side, height=6, wrap="word", fg="#b00000", relief="flat",
                                 font=("Segoe UI", 9))
        self.flags_txt.pack(fill=tk.X)
        legend = ttk.Label(side, foreground="#555", font=("Segoe UI", 8), text=(
            "green = counted   yellow = added by you   red X = removed\n"
            "orange ring = doubtful (check)   white ring = filled-in gap"))
        legend.pack(anchor="w", pady=(2, 0))

        form = ttk.LabelFrame(side, text="Part", padding=6)
        form.pack(fill=tk.X, pady=6)
        self.part = tk.StringVar()
        self.operator = tk.StringVar()
        self.expected = tk.StringVar()
        self.layers = tk.IntVar(value=1)
        self.toroid = tk.StringVar(value=self.presets["toroids"][0]["name"])
        self.wire = tk.StringVar(value=self.presets["wires"][0]["name"])
        self.recipe_txt = tk.StringVar(value="")
        rows = [("Part / lot", ttk.Entry(form, textvariable=self.part)),
                ("Operator", ttk.Entry(form, textvariable=self.operator)),
                ("Toroid", ttk.Combobox(form, textvariable=self.toroid, state="readonly",
                                        values=[t["name"] for t in self.presets["toroids"]])),
                ("Wire", ttk.Combobox(form, textvariable=self.wire, state="readonly",
                                      values=[w["name"] for w in self.presets["wires"]])),
                ("Layers", ttk.Spinbox(form, from_=1, to=5, width=4, textvariable=self.layers)),
                ("Expected turns", ttk.Entry(form, textvariable=self.expected)),
                ("Recipe", ttk.Combobox(form, textvariable=self.recipe_txt,
                                        values=self.presets.get("recipes", [""])))]
        for r, (lbl, w) in enumerate(rows):
            ttk.Label(form, text=lbl).grid(row=r, column=0, sticky="w", pady=2)
            w.grid(row=r, column=1, sticky="ew", pady=2)
        form.columnconfigure(1, weight=1)
        rc = rows[-1][1]
        self.recipe_box = rc
        ttk.Button(form, text="Build...", width=8, command=self.open_builder).grid(
            row=len(rows) - 1, column=2, padx=(4, 0))
        self.refresh_recipes()
        rc.bind("<<ComboboxSelected>>", lambda e: self.recount())
        rc.bind("<Return>", lambda e: self.recount())
        rc.bind("<FocusOut>", lambda e: self.update_side())
        self.expected.trace_add("write", lambda *a: self.update_side())

        ttk.Button(side, text="Confirm & log  [Enter]", style="Big.TButton",
                   command=self.confirm).pack(fill=tk.X, ipady=8, pady=4)
        self.learn = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Save confirmed parts for learning", variable=self.learn).pack(anchor="w")
        ttk.Button(side, text="Open log folder", command=self.open_log).pack(fill=tk.X, pady=(4, 0))
        self.status = ttk.Label(self, relief="sunken", anchor="w",
                                text="Left-click: add turn   Right-click: remove   Ctrl+Z: undo   R: set ring")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        def not_typing():
            w = self.focus_get()
            return not isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text))
        self.bind("<space>", lambda e: not_typing() and self.capture())
        self.bind("<Return>", lambda e: not_typing() and self.confirm())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Escape>", lambda e: self.cancel_ring_mode())
        self.bind("r", lambda e: not_typing() and self.start_ring_mode())
        self.bind("R", lambda e: not_typing() and self.start_ring_mode())

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        try:
            s = json.loads(settings_path().read_text(encoding="utf-8"))
        except Exception:
            return
        for key, var in (("operator", self.operator), ("toroid", self.toroid), ("wire", self.wire),
                         ("recipe", self.recipe_txt), ("expected", self.expected), ("camera", self.cam_choice)):
            if s.get(key) is not None:
                var.set(s[key])
        self.keep_ring.set(bool(s.get("keep_ring", False)))
        self.learn.set(bool(s.get("learn", True)))
        self._urls = s.get("urls", [])
        if s.get("ring"):
            try:
                self._saved_ring = core.Ring(**s["ring"])
            except Exception:
                self._saved_ring = None

    def _save_settings(self):
        s = {"operator": self.operator.get(), "toroid": self.toroid.get(), "wire": self.wire.get(),
             "recipe": self.recipe_txt.get(), "expected": self.expected.get(), "camera": self.cam_choice.get(),
             "keep_ring": self.keep_ring.get(), "learn": self.learn.get(), "urls": getattr(self, "_urls", []),
             "ring": self.ring.to_dict() if (self.ring and self.keep_ring.get()) else None}
        try:
            settings_path().parent.mkdir(parents=True, exist_ok=True)
            settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------ cameras
    def refresh_cameras(self):
        self.status.config(text="Looking for cameras...")
        self.update_idletasks()
        cams = cameras.list_cameras()
        values = [cameras.label(i, n) for i, n in cams] + list(getattr(self, "_urls", [])) + [cameras.URL_HINT]
        self.cam_box.config(values=values)
        if self.cam_choice.get() not in values:
            self.cam_choice.set(values[0] if cams else cameras.URL_HINT)
        self.status.config(text=f"{len(cams)} camera(s) found. Pick one and press Start camera.")

    def on_camera_choice(self, _e=None):
        if self.cam_choice.get() != cameras.URL_HINT:
            return
        url = simpledialog.askstring(
            "Network / phone camera",
            "Stream URL, for example:\n"
            "  http://192.168.1.50:8080/video   (phone 'IP Webcam' app, MJPEG)\n"
            "  rtsp://192.168.1.60:554/stream1  (IP / RTSP camera)", parent=self)
        if not url:
            self.cam_choice.set(self.cam_box.cget("values")[0])
            return
        self._urls = [url] + [u for u in getattr(self, "_urls", []) if u != url][:4]
        vals = list(self.cam_box.cget("values"))
        if url not in vals:
            self.cam_box.config(values=[v for v in vals if v != cameras.URL_HINT] + [url, cameras.URL_HINT])
        self.cam_choice.set(url)

    def toggle_camera(self):
        if self.cap is not None:
            self.stop_camera()
            return
        src = cameras.parse_label(self.cam_choice.get())
        if src is None:
            messagebox.showinfo("Camera", "Pick a camera from the list first.")
            return
        self.status.config(text=f"Opening {self.cam_choice.get()}...")
        self.update_idletasks()
        try:
            self.cap = cameras.open_source(src)
        except Exception as e:
            messagebox.showerror("Camera", f"{e}\n\nCheck the cable / URL, or close other apps using the camera.")
            self.status.config(text="Camera not opened.")
            return
        self.cam_btn.config(text="Stop camera")
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.config(text=f"Live: {self.cam_choice.get()}  {w}x{h}.  Position the core, then Capture [Space].")
        self.result, self.marks = None, []
        if not self.keep_ring.get():
            self.ring = None
        self._live_tick = 0
        self.live_loop()

    def stop_camera(self):
        if self.live_job:
            self.after_cancel(self.live_job)
            self.live_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cam_btn.config(text="Start camera")

    def live_loop(self):
        ok, frame = self.cap.read() if self.cap else (False, None)
        if ok:
            self.img = frame
            self._live_tick += 1
            if self.live_count.get() and self._live_tick % 10 == 0:
                self.recount(quiet=True)
            else:
                self.redraw()
        self.live_job = self.after(40, self.live_loop)

    def capture(self):
        if self.cap is None or self.img is None:
            if self.cam_choice.get() and cameras.parse_label(self.cam_choice.get()) is not None and self.cap is None:
                self.toggle_camera()
                return
            messagebox.showinfo("Capture", "Start the camera first.")
            return
        frame = self.img.copy()
        self.stop_camera()
        self.cam_was_on = True
        self.set_image(frame, keep_ring=self.keep_ring.get())

    # ------------------------------------------------------------------ input
    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not p:
            return
        self.stop_camera()
        self.cam_was_on = False
        data = np.fromfile(p, dtype=np.uint8)  # works with non-ASCII Windows paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read this image.")
            return
        self.set_image(img, keep_ring=self.keep_ring.get())

    def set_image(self, img, keep_ring=False):
        self.img = img
        if not keep_ring:
            self.ring = None
        elif self.ring is None and getattr(self, "_saved_ring", None):
            self.ring = self._saved_ring
        self.recount()

    # ------------------------------------------------------------------ counting
    def params(self) -> core.CountParams:
        t = next((x for x in self.presets["toroids"] if x["name"] == self.toroid.get()), {})
        w = next((x for x in self.presets["wires"] if x["name"] == self.wire.get()), {})
        rec, _ = self.current_recipe()
        colors = {c for s in rec for c in s.turns} if rec else set()
        return core.CountParams(toroid_id_mm=t.get("id_mm"), wire_od_mm=w.get("od_mm"),
                                expected_layers=int(self.layers.get() or 1),
                                sections=len(rec) if rec else 1, multi_color=len(colors) > 1)

    def refresh_recipes(self):
        saved = recipe.load_saved(recipes_path())["recipes"]
        base = [r for r in self.presets.get("recipes", [""]) if r not in saved]
        self.recipe_box["values"] = [""] + saved + [r for r in base if r]

    def open_builder(self):
        names = list(dict.fromkeys(list(recipe.DEFAULT_COLORS) + list(self.presets.get("colors", {}))))
        names.remove("copper")
        names.insert(0, "copper")

        def use(txt):
            self.recipe_txt.set(txt)
            self.refresh_recipes()
            self._save_settings()
            self.recount()
        builder.RecipeBuilder(self, recipes_path(), names, use, self.recipe_txt.get())

    def current_recipe(self):
        txt = self.recipe_txt.get().strip()
        if not txt:
            return None, ""
        try:
            return recipe.parse(txt), ""
        except ValueError as e:
            return None, str(e)

    def recipe_eval(self):
        """(lines, {angle: dot colour}, border angles) for the current marks, or (None, {}, [])."""
        rec, _ = self.current_recipe()
        if not rec or not self.ring or not self.marks or self.img is None:
            return None, {}, []
        refs = {k: tuple(v) for k, v in self.presets.get("colors", {}).items()}
        lines, cols, angs = recipe.evaluate(self.img, self.ring, self.marks, rec, refs)
        dots = {a: recipe.DOT_BGR.get(c, (160, 160, 160)) for a, c in zip(angs, cols)}
        return lines, dots, recipe.gap_angles(angs, len(rec))

    def recount(self, quiet=False):
        if self.img is None:
            return
        try:
            res = core.count_turns(self.img, ring=self.ring, params=self.params())
        except Exception as e:
            self.result, self.marks = None, []
            if not quiet:
                self.status.config(text=f"{e}")
                self.start_ring_mode()
            self.update_side()
            self.redraw()
            return
        if self.use_yolo.get() and yolo_backend.available():
            try:
                res.angles = yolo_backend.detect_angles(self.img, res.ring)
                res.count = len(res.angles)
                res.flags.insert(0, "counted with YOLO model")
            except Exception as e:
                res.flags.insert(0, f"YOLO failed, OpenCV used: {e}")
        self.result = res
        try:
            self.meas = measure.measure(self.img, res.ring, res.count)
        except Exception:
            self.meas = None
        self.meas_lbl.config(text=measure.summary(self.meas) if self.meas else "")
        self.ring = res.ring
        self.marks = list(res.angles)
        self.added, self.removed, self.undo_stack = [], [], []
        self.update_side()
        self.redraw()
        if not quiet:
            self.status.config(text="Check the lines: left-click adds a missed turn, right-click removes a wrong one. "
                                    "Then Confirm & log [Enter].")

    def reset_marks(self):
        if self.result:
            self.marks = list(self.result.angles)
            self.added, self.removed, self.undo_stack = [], [], []
            self.update_side()
            self.redraw()

    def undo(self):
        if not self.undo_stack:
            return
        op, a = self.undo_stack.pop()
        if op == "add":
            self.marks.remove(a)
            self.added.remove(a)
        else:
            self.marks.append(a)
            if a in self.removed:
                self.removed.remove(a)
            else:  # it was a manual mark that got removed
                self.added.append(a)
        self.update_side()
        self.redraw()

    # ------------------------------------------------------------------ mouse
    def to_img(self, ev):
        return ((ev.x - self.offset[0]) / self.scale, (ev.y - self.offset[1]) / self.scale)

    def start_ring_mode(self):
        if self.img is None:
            return
        if self.cap is not None:  # freeze the picture while the ring is being set
            self.capture()
        self.mode = "ring"
        self.ring_clicks = []
        self.status.config(text="SET RING - " + RING_STEPS[0] + "     (right-click = step back, Esc = cancel)")
        self.redraw()

    def cancel_ring_mode(self):
        if self.mode == "ring":
            self.mode = "edit"
            self.ring_clicks = []
            self.status.config(text="Ring setting cancelled.")
            self.redraw()

    def on_motion(self, ev):
        if self.mode == "ring" and self.img is not None:
            self.mouse = self.to_img(ev)
            self.redraw()

    def on_left(self, ev):
        if self.img is None:
            return
        x, y = self.to_img(ev)
        if self.mode == "ring":
            self.ring_clicks.append((x, y))
            if len(self.ring_clicks) == 3:
                c, i, o = self.ring_clicks
                if math.dist(c, o) < math.dist(c, i):  # clicked OD before ID: just swap
                    i, o = o, i
                self.ring = core.ring_from_points(c, i, o, like=self.ring)
                self.mode = "edit"
                self.recount()
                if self.keep_ring.get():
                    self._save_settings()
            else:
                self.status.config(text="SET RING - " + RING_STEPS[len(self.ring_clicks)] +
                                   "     (right-click = step back, Esc = cancel)")
                self.redraw()
            return
        if not self.ring or not self.result:
            return
        a = self.ring.angle_of(x, y)
        self.marks.append(a)
        self.added.append(a)
        self.undo_stack.append(("add", a))
        self.update_side()
        self.redraw()

    def on_right(self, ev):
        if self.mode == "ring":
            if self.ring_clicks:
                self.ring_clicks.pop()
                self.status.config(text="SET RING - " + RING_STEPS[len(self.ring_clicks)])
                self.redraw()
            return
        if self.img is None or not self.ring or not self.marks:
            return
        x, y = self.to_img(ev)
        a = self.ring.angle_of(x, y)
        d = [abs(math.atan2(math.sin(a - m), math.cos(a - m))) for m in self.marks]
        i = int(np.argmin(d))
        m = self.marks.pop(i)
        if m in self.added:
            self.added.remove(m)
        else:
            self.removed.append(m)
        self.undo_stack.append(("remove", m))
        self.update_side()
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def overlay(self, with_status=False):
        if self.img is None:
            return None
        if self.ring is None or self.result is None:
            ov = self.img.copy()
            if self.ring is not None:  # e.g. live view with a kept ring: show where it is
                ov = core.draw_overlay(ov, self.ring, [], lines=False)
            return ov
        auto = [m for m in self.marks if m not in self.added]
        _, dots, borders = self.recipe_eval()
        doubtful = [a for a in (self.result.doubtful if self.result else []) if a in self.marks]
        ov = core.draw_overlay(self.img, self.ring, auto, removed=self.removed, added=self.added,
                               dot_colors=dots, borders=borders, doubtful=doubtful)
        for a in (self.result.gap_added if self.result else []):
            if a in self.marks:
                r = self.ring.r_in + 0.45 * (self.ring.r_out - self.ring.r_in)
                x, y = self.ring.pt(a, r)
                cv2.circle(ov, (int(x), int(y)), 9, (255, 255, 255), 2, cv2.LINE_AA)
        if with_status:  # burn the result into the saved overlay
            txt = f"{len(self.marks)} turns  {self.pass_lbl.cget('text')}"
            cv2.putText(ov, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(ov, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return ov

    def _draw_ring_guide(self, ov):
        """Guided ring setting: what is clicked so far + live preview under the mouse."""
        ov = ov.copy()
        t = max(1, int(round(max(ov.shape[:2]) / 700)))
        clicks = self.ring_clicks
        m = self.mouse
        col = (255, 0, 255)
        if clicks:
            c = clicks[0]
            cv2.drawMarker(ov, (int(c[0]), int(c[1])), col, cv2.MARKER_CROSS, 30 * t, 2 * t)
            if len(clicks) >= 2:
                cv2.circle(ov, (int(c[0]), int(c[1])), int(math.dist(c, clicks[1])), (0, 255, 255), 2 * t,
                           cv2.LINE_AA)
            if m is not None:
                cv2.circle(ov, (int(c[0]), int(c[1])), int(math.dist(c, m)), col, t, cv2.LINE_AA)
        step = RING_STEPS[min(len(clicks), 2)]
        h = 40 * t
        cv2.rectangle(ov, (0, 0), (ov.shape[1], h), (40, 0, 40), -1)
        cv2.putText(ov, step, (10 * t, 28 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (255, 255, 255), 2 * t, cv2.LINE_AA)
        return ov

    def redraw(self):
        self.canvas.delete("all")
        ov = self.overlay()
        if ov is None:
            self.canvas.create_text(20, 20, anchor="nw", fill="#bbb", font=("Segoe UI", 11), text=HELP)
            return
        if self.mode == "ring":
            ov = self._draw_ring_guide(ov)
        cw, ch = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        h, w = ov.shape[:2]
        self.scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        self.offset = ((cw - nw) // 2, (ch - nh) // 2)
        rgb = cv2.cvtColor(cv2.resize(ov, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(*self.offset, anchor="nw", image=self._tk)

    def update_side(self):
        r = self.result
        n = len(self.marks) if r else None
        self.count_lbl.config(text=str(n) if n is not None else "--")
        self.flags_txt.delete("1.0", tk.END)
        if r:
            edits = f"   (auto {r.count}, +{len(self.added)} / -{len(self.removed)})" if (self.added or self.removed) else ""
            self.conf_lbl.config(text=f"Confidence {r.confidence:.0%}{edits}")
            self.flags_txt.insert(tk.END, "\n".join("! " + f for f in r.flags) or "No warnings.")
        self.pass_lbl.config(text="", fg="black")
        self.table.delete(*self.table.get_children())
        rec, err = self.current_recipe()
        if err:
            self.flags_txt.insert("1.0", "Recipe: " + err + "\n")
        lines, _, _ = self.recipe_eval()
        self.lines = lines
        if lines:
            for ln in lines:
                self.table.insert("", tk.END, values=(ln.section, ln.color, ln.expected, ln.counted,
                                                      "OK" if ln.ok else f"{ln.counted - ln.expected:+d}"),
                                  tags=("good" if ln.ok else "bad",))
            bad = [ln for ln in lines if not ln.ok]
            if not bad:
                self.pass_lbl.config(text="PASS (recipe)", fg="#008000")
            elif all(ln.section == "?" for ln in bad):
                self.pass_lbl.config(text="CHECK grey marks", fg="#c07000")
            else:
                self.pass_lbl.config(text="FAIL (recipe)", fg="#c00000")
            return
        exp = self.expected.get().strip()
        if r and exp.isdigit():
            ok = n == int(exp)
            self.pass_lbl.config(text="PASS" if ok else f"FAIL ({n - int(exp):+d})",
                                 fg="#008000" if ok else "#c00000")

    # ------------------------------------------------------------------ log
    def confirm(self):
        if not self.result or self.img is None:
            messagebox.showinfo("Confirm", "Nothing to confirm yet. Capture or open an image first.")
            return
        open_doubts = [a for a in self.result.doubtful if a in self.marks]
        warn = list(self.result.flags)
        if warn and not messagebox.askyesno(
                "Confirm", "Warnings:\n\n" + "\n".join(warn) +
                ("\n\nOrange-ringed marks are still in the count." if open_doubts else "") +
                "\n\nDid you check the lines on the image and want to log this count?"):
            return
        n = len(self.marks)
        exp = self.expected.get().strip()
        row = {"part": self.part.get(), "operator": self.operator.get(), "toroid": self.toroid.get(),
               "wire": self.wire.get(), "layers": self.layers.get(), "expected": exp,
               "auto_count": self.result.count, "final_count": n, "added": len(self.added),
               "removed": len(self.removed),
               "result": self.pass_lbl.cget("text"),
               "recipe": self.recipe_txt.get().strip(),
               "sections": " | ".join(f"{ln.section} {ln.color} {ln.counted}/{ln.expected}"
                                      for ln in (self.lines or [])),
               "confidence": round(self.result.confidence, 3), "flags": self.result.flags,
               "ring": self.ring.to_dict()}
        if self.meas:
            row.update(od_mm=round(self.meas["od_mm"], 2), id_mm=round(self.meas["id_mm"], 2),
                       inner_pitch_mm=round(self.meas.get("inner_pitch_mm", 0), 3))
        try:
            p = logbook.log_result(logbook.default_dir(), self.img, self.overlay(with_status=True), row)
            if self.learn.get():
                logbook.save_sample(logbook.default_dir(), self.img, self.ring, self.marks, row)
        except Exception as e:
            messagebox.showerror("Log", f"Could not save the log: {e}")
            return
        self._save_settings()
        self.status.config(text=f"Logged {n} turns ({self.pass_lbl.cget('text') or 'no target'}) to {p}")
        if self.cam_was_on:  # fixed station: go straight back to live view for the next part
            self.after(600, self.toggle_camera)

    def open_log(self):
        d = logbook.default_dir()
        d.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            import os
            os.startfile(d)  # noqa: S606
        else:
            messagebox.showinfo("Log folder", str(d))

    def on_close(self):
        self._save_settings()
        self.stop_camera()
        self.destroy()


def main():
    App().mainloop()
