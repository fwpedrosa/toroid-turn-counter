"""Recipe builder window: pick a layout (or make your own), fill turns per section, save by name."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from . import recipe

CUSTOM = "Custom..."


class RecipeBuilder(tk.Toplevel):
    """on_use(text) is called with the recipe text when the user presses Use or Save."""

    def __init__(self, master, store: Path, colors: list[str], on_use, current: str = ""):
        super().__init__(master)
        self.title("Recipe builder")
        self.transient(master)
        self.resizable(False, False)
        self.store = Path(store)
        self.colors = colors or ["copper"]
        self.on_use = on_use
        self.rows: list[dict] = []
        self.shares: list[float] = [1.0]

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Layout").grid(row=0, column=0, sticky="w")
        self.layout = tk.StringVar()
        self.layout_box = ttk.Combobox(top, textvariable=self.layout, state="readonly", width=28)
        self.layout_box.grid(row=0, column=1, sticky="w", padx=4)
        self.layout_box.bind("<<ComboboxSelected>>", lambda e: self.on_layout())

        self.custom = ttk.Frame(top)
        self.custom.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(self.custom, text="Share of the core per section").pack(side=tk.LEFT)
        self.share_txt = tk.StringVar(value="1/4, 3/4")
        e = ttk.Entry(self.custom, textvariable=self.share_txt, width=18)
        e.pack(side=tk.LEFT, padx=4)
        e.bind("<Return>", lambda ev: self.apply_custom())
        ttk.Button(self.custom, text="Apply", command=self.apply_custom).pack(side=tk.LEFT)
        ttk.Button(self.custom, text="Save layout as...", command=self.save_layout).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, foreground="#555", font=("Segoe UI", 8),
                  text="Custom examples:  1/4, 3/4    1/2, 1/4, 1/4    1 2 1  (parts)    40%, 60%").grid(
            row=2, column=0, columnspan=3, sticky="w")

        self.grid_fr = ttk.LabelFrame(self, text="Sections", padding=8)
        self.grid_fr.pack(fill=tk.X, padx=8)

        self.preview = tk.StringVar()
        pv = ttk.LabelFrame(self, text="Recipe text", padding=6)
        pv.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(pv, textvariable=self.preview, wraplength=520, foreground="#004080").pack(anchor="w")

        btns = ttk.Frame(self, padding=(8, 0, 8, 8))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Same as A for all", command=self.same_for_all).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Save recipe as...", command=self.save_recipe).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Use", command=self.use).pack(side=tk.RIGHT)

        self.refresh_layouts()
        self.load_text(current)
        self.grab_set()
        self.focus_set()

    # ------------------------------------------------------------------ layouts
    def all_layouts(self) -> dict:
        saved = recipe.load_saved(self.store)["layouts"]
        return {**recipe.LAYOUTS, **saved}

    def refresh_layouts(self):
        names = list(self.all_layouts()) + [CUSTOM]
        self.layout_box["values"] = names

    def on_layout(self):
        name = self.layout.get()
        if name == CUSTOM:
            self.custom.grid()
            self.apply_custom()
            return
        self.custom.grid_remove()
        self.set_shares(self.all_layouts().get(name, [1.0]))

    def apply_custom(self):
        try:
            self.set_shares(recipe.parse_shares(self.share_txt.get()))
        except ValueError as e:
            messagebox.showerror("Layout", str(e), parent=self)

    def save_layout(self):
        try:
            shares = recipe.parse_shares(self.share_txt.get())
        except ValueError as e:
            messagebox.showerror("Layout", str(e), parent=self)
            return
        name = simpledialog.askstring("Save layout", "Name for this layout (e.g. 1/4 + 3/4):", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        if name in recipe.LAYOUTS or name == CUSTOM:
            messagebox.showerror("Layout", "That name is a built-in layout. Pick another.", parent=self)
            return
        recipe.save_layout(self.store, name, shares)
        self.refresh_layouts()
        self.layout.set(name)
        self.custom.grid_remove()
        self.set_shares(shares)

    # ------------------------------------------------------------------ section rows
    def set_shares(self, shares: list[float], keep=True):
        old = [self.row_values(r) for r in self.rows] if keep else []
        self.shares = list(shares)
        for w in self.grid_fr.winfo_children():
            w.destroy()
        self.rows = []
        for c, t in enumerate(["Section", "Share", "Turns", "Bifilar", "Wire colour", "2nd colour"]):
            ttk.Label(self.grid_fr, text=t, font=("Segoe UI", 9, "bold")).grid(row=0, column=c, padx=4, sticky="w")
        for i, sh in enumerate(self.shares):
            name = chr(ord("A") + i)
            r = {"name": name,
                 "turns": tk.StringVar(value="10"),
                 "bif": tk.BooleanVar(value=False),
                 "c1": tk.StringVar(value=self.colors[0]),
                 "c2": tk.StringVar(value=self._second(self.colors[0]))}
            if i < len(old):
                self.put_row(r, old[i])
            ttk.Label(self.grid_fr, text=name).grid(row=i + 1, column=0, padx=4)
            ttk.Label(self.grid_fr, text=recipe.frac_text(sh)).grid(row=i + 1, column=1, padx=4)
            ttk.Spinbox(self.grid_fr, from_=1, to=999, width=6, textvariable=r["turns"]).grid(row=i + 1, column=2, padx=4)
            ttk.Checkbutton(self.grid_fr, variable=r["bif"], command=self.update).grid(row=i + 1, column=3)
            ttk.Combobox(self.grid_fr, textvariable=r["c1"], values=self.colors, width=10).grid(row=i + 1, column=4, padx=4)
            r["c2_box"] = ttk.Combobox(self.grid_fr, textvariable=r["c2"], values=self.colors, width=10)
            r["c2_box"].grid(row=i + 1, column=5, padx=4)
            for k in ("turns", "c1", "c2"):
                r[k].trace_add("write", lambda *a: self.update())
            self.rows.append(r)
        ttk.Label(self.grid_fr, foreground="#555", font=("Segoe UI", 8),
                  text="Bifilar: two wires wound together, each with the turns shown (e.g. red 13 + green 13).").grid(
            row=len(self.shares) + 1, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self.update()

    def _second(self, c1: str) -> str:
        for c in ("green", "red", *self.colors):
            if c != c1 and c in self.colors:
                return c
        return c1

    @staticmethod
    def row_values(r) -> dict:
        return {"turns": r["turns"].get(), "bif": r["bif"].get(), "c1": r["c1"].get(), "c2": r["c2"].get()}

    @staticmethod
    def put_row(r, v: dict):
        r["turns"].set(v["turns"])
        r["bif"].set(v["bif"])
        r["c1"].set(v["c1"])
        r["c2"].set(v["c2"])

    def same_for_all(self):
        if self.rows:
            v = self.row_values(self.rows[0])
            for r in self.rows[1:]:
                self.put_row(r, v)
            self.update()

    # ------------------------------------------------------------------ text
    def sections(self) -> list[recipe.Section]:
        out = []
        for r, sh in zip(self.rows, self.shares):
            t = r["turns"].get().strip()
            if not t.isdigit() or int(t) < 1:
                raise ValueError(f"Section {r['name']}: turns must be a whole number above 0.")
            c1 = r["c1"].get().strip().lower() or "copper"
            turns = {c1: int(t)}
            if r["bif"].get():
                c2 = r["c2"].get().strip().lower()
                if not c2 or c2 == c1:
                    raise ValueError(f"Section {r['name']}: bifilar needs two different colours.")
                turns[c2] = int(t)
            out.append(recipe.Section(r["name"], turns, sh))
        return out

    def text(self, title: str = "") -> str:
        return recipe.to_text(self.sections(), title)

    def update(self):
        for r in self.rows:
            r["c2_box"].state(["!disabled"] if r["bif"].get() else ["disabled"])
        try:
            self.preview.set(self.text())
        except ValueError as e:
            self.preview.set(str(e))

    def load_text(self, txt: str):
        """Fill the window from an existing recipe, so it can be edited."""
        try:
            secs = recipe.parse(txt) if txt.strip() else []
        except ValueError:
            secs = []
        if not secs:
            self.layout.set("Halves (1/2 + 1/2)")
            self.custom.grid_remove()
            self.set_shares(recipe.LAYOUTS["Halves (1/2 + 1/2)"], keep=False)
            return
        n = len(secs)
        shares = [s.share for s in secs]
        if any(x is None for x in shares):
            shares = [1 / n] * n
        match = next((k for k, v in self.all_layouts().items()
                      if len(v) == n and all(abs(a - b) < 1e-3 for a, b in zip(v, shares))), None)
        if match:
            self.layout.set(match)
            self.custom.grid_remove()
        else:
            self.layout.set(CUSTOM)
            self.share_txt.set(", ".join(recipe.frac_text(x) for x in shares))
            self.custom.grid()
        self.set_shares(shares, keep=False)
        for r, s in zip(self.rows, secs):
            cols = list(s.turns.items())
            r["turns"].set(str(cols[0][1]))
            r["c1"].set(cols[0][0])
            if len(cols) > 1:
                r["bif"].set(True)
                r["c2"].set(cols[1][0])
        self.update()

    # ------------------------------------------------------------------ actions
    def use(self, title: str = ""):
        try:
            txt = self.text(title)
        except ValueError as e:
            messagebox.showerror("Recipe", str(e), parent=self)
            return None
        self.on_use(txt)
        return txt

    def save_recipe(self):
        try:
            self.text()
        except ValueError as e:
            messagebox.showerror("Recipe", str(e), parent=self)
            return
        name = simpledialog.askstring("Save recipe", "Name for this recipe (e.g. Choke 3x18):", parent=self)
        if not name or not name.strip():
            return
        name = name.strip().replace("[", "(").replace("]", ")")
        existing = [recipe.strip_name(r)[0] for r in recipe.load_saved(self.store)["recipes"]]
        if name in existing and not messagebox.askyesno(
                "Save recipe", f"A recipe called '{name}' exists. Replace it?", parent=self):
            return
        txt = self.text(name)
        recipe.save_recipe(self.store, txt)
        self.on_use(txt)
        self.destroy()
