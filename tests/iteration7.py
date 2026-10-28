"""Lancer dans PyCharm pour tester Z mono avec l'étalonnage (data/mono.npz; touche c : étalonner R)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--iteration", "7"])
