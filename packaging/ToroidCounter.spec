# PyInstaller spec: builds dist/ToroidCounter.exe (one file, no console window).
# Run from the repo root:  pyinstaller packaging/ToroidCounter.spec --noconfirm
from pathlib import Path

root = Path(SPECPATH).parent
datas = [(str(root / "toroid_counter" / "presets.json"), "."),
         (str(root / "LICENSE"), ".")]
if (root / "models" / "turns.pt").exists():  # optional YOLO model
    datas.append((str(root / "models" / "turns.pt"), "models"))

hidden = []
try:  # camera names on Windows (optional)
    from PyInstaller.utils.hooks import collect_submodules
    hidden = collect_submodules("pygrabber") + collect_submodules("comtypes")
except Exception:
    pass

a = Analysis([str(root / "app.py")], pathex=[str(root)], datas=datas, hiddenimports=hidden,
             excludes=["matplotlib", "torch", "ultralytics", "pandas", "IPython", "pytest"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="ToroidCounter",
          console=False, upx=False, icon=None)
