"""Expérience 6 : comptage des doigts sur de VRAIES mains (masques enregistrés).
"""
import cv2
import numpy as np

from common import ROOT, plt, save
from g3d.config import SegmentationConfig
from g3d.hand import extract_hand
from test_iter2_doigts import classe


def run():
    folder = ROOT / "data" / "real_hands"
    files = sorted(p for p in folder.glob("*.png") if not p.stem.endswith("_img") and p.stem[:1].isdigit()) \
        if folder.exists() else []
    if not files:
        return {"disponible": False,
                "message": "aucune donnée : lancer scripts/record_dataset.py pour enregistrer des mains"}
    conf = np.zeros((6, 7), int)
    exact = utiles = 0
    for p in files:
        att = int(p.stem[0])
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        obs = extract_hand((m > 127).astype(np.uint8) * 255, SegmentationConfig())
        lu = obs.fingers if obs is not None else 6
        conf[min(att, 5), lu] += 1
        exact += lu == att
        utiles += lu != 6 and classe(lu) == classe(att)
    n = len(files)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(conf, cmap="Greens")
    ax.grid(False)
    for i in range(6):
        for j in range(7):
            if conf[i, j]:
                ax.text(j, i, conf[i, j], ha="center", va="center")
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(i) for i in range(6)] + ["aucune"])
    ax.set_yticks(range(6))
    ax.set_xlabel("doigts lus")
    ax.set_ylabel("doigts montrés")
    ax.set_title(f"Mains réelles ({n}) : {100 * exact / n:.1f} % exact, classes {100 * utiles / n:.1f} %")
    f = save(fig, "fig8_mains_reelles.png")
    return {"disponible": True, "images": n, "exact_pct": round(100 * exact / n, 2),
            "classes_utiles_pct": round(100 * utiles / n, 2), "figures": [f]}


if __name__ == "__main__":
    print(run())
