"""Chemins et style communs aux expériences (graines fixes, figures dans results/)."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                  # pas de fenêtre : figures enregistrées
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
for p in (ROOT, ROOT / "experiments", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
BLEU, ORANGE, GRIS = "#2a6fb0", "#d9792b", "#8a8a8a"


def save(fig, name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = RESULTS / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(ROOT))
