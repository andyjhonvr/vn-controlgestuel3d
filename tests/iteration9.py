"""Lancer dans PyCharm pour régénérer les 7 figures et results/summary.json (sans caméra)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "experiments" / "run_all.py"), run_name="__main__")
