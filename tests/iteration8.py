"""Lancer dans PyCharm pour tester la stéréo 3D (2e caméra par défaut : --cam2 1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--cam2" not in args:
        args += ["--cam2", "1"]
    main([*args, "--iteration", "8"])
