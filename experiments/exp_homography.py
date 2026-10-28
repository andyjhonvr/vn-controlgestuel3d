"""Expérience 3 : précision de la zone de contrôle (homographie de la feuille lettre).
"""
import cv2
import numpy as np

from common import BLEU, ORANGE, plt, save
from g3d.geometry import apply_homography, dlt_homography
from synth import letter_sheet_homography


def run():
    sigmas = [0, 0.5, 1, 2, 3, 4, 5]
    rng = np.random.default_rng(31)
    dlt, ocv, diff = [], [], []
    for s in sigmas:
        e1, e2, d = [], [], []
        for _ in range(300):
            sheet, img, (w, h) = letter_sheet_homography(rng)
            clicks = img + rng.normal(0, s, img.shape)
            H_true = cv2.getPerspectiveTransform(sheet.astype(np.float32), img.astype(np.float32))
            uv = rng.uniform([0, 0], [w, h], (30, 2))
            px = apply_homography(H_true, uv)
            Hm = dlt_homography(clicks, sheet)
            Hc, _ = cv2.findHomography(clicks, sheet, 0)
            a, b = apply_homography(Hm, px), apply_homography(Hc, px)
            e1.append(np.linalg.norm(a - uv, axis=1).mean())
            e2.append(np.linalg.norm(b - uv, axis=1).mean())
            d.append(np.abs(a - b).max())
        dlt.append(float(np.mean(e1)))
        ocv.append(float(np.mean(e2)))
        diff.append(float(np.max(d)))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(sigmas, dlt, "o-", color=BLEU, label="DLT normalisée (maison)")
    ax.plot(sigmas, ocv, "x--", color=ORANGE, label="cv2.findHomography")
    ax.set_xlabel("bruit sur les clics σ (px)")
    ax.set_ylabel("erreur moyenne sur la feuille (cm)")
    ax.set_title("Zone de contrôle : feuille lettre (4 coins cliqués)")
    ax.legend()
    f = save(fig, "fig4_homographie.png")
    return {"erreur_cm": {f"sigma_{s}px": round(e, 3) for s, e in zip(sigmas, dlt)},
            "ecart_max_dlt_opencv_cm": float(max(diff)), "figures": [f]}


if __name__ == "__main__":
    print(run())
