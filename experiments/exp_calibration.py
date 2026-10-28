"""Expérience 2 : étalonnage par damier sur images synthétiques (K et distorsion connus).
"""
import numpy as np

from common import BLEU, ORANGE, plt, save
from g3d.calibration import calibrate_mono, find_corners
from synth import DIST, camera_K, random_board_poses, render_board

PATTERN, SQ, F = (9, 6), 2.5, 600.0


def essai(n_vues, noise, blur, seed):
    rng = np.random.default_rng(seed)
    K = camera_K(F)
    pts = []
    for rv, tv in random_board_poses(n_vues, rng, PATTERN, SQ):
        c = find_corners(render_board(K, DIST, rv, tv, PATTERN, SQ, noise=noise, blur=blur, rng=rng), PATTERN)
        if c is not None:
            pts.append(c)
    r = calibrate_mono(pts, (640, 480), PATTERN, SQ)
    return r["rms"], 100 * (r["K"][0, 0] / F - 1)


def run():
    vues = [5, 10, 15, 20]
    res = {}
    for nom, noise, blur in (("nominal", 2.0, 0.8), ("difficile", 8.0, 1.6)):
        rms, fx = [], []
        for n in vues:
            e = [essai(n, noise, blur, 100 + s) for s in range(5)]
            rms.append(float(np.mean([a for a, _ in e])))
            fx.append(float(np.mean([abs(b) for _, b in e])))
        res[nom] = {"vues": vues, "rms_px": rms, "fx_erreur_abs_pct": fx}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for nom, col in (("nominal", BLEU), ("difficile", ORANGE)):
        axs[0].plot(vues, res[nom]["rms_px"], "o-", color=col, label=nom)
        axs[1].plot(vues, res[nom]["fx_erreur_abs_pct"], "o-", color=col, label=nom)
    axs[0].axhline(0.5, color="red", ls="--", lw=1, label="critère 0,5 px")
    axs[0].set_xlabel("nombre de vues")
    axs[0].set_ylabel("RMS de reprojection (px)")
    axs[0].set_title("Erreur de reprojection")
    axs[0].legend()
    axs[1].set_xlabel("nombre de vues")
    axs[1].set_ylabel("|erreur sur fx| (%)")
    axs[1].set_title("Focale estimée (f vraie = 600 px)")
    axs[1].legend()
    f = save(fig, "fig3_etalonnage.png")
    i15 = vues.index(15)
    return {"15_vues_nominal": {"rms_px": round(res["nominal"]["rms_px"][i15], 4),
                                "fx_erreur_pct": round(res["nominal"]["fx_erreur_abs_pct"][i15], 4)},
            "15_vues_difficile": {"rms_px": round(res["difficile"]["rms_px"][i15], 4),
                                  "fx_erreur_pct": round(res["difficile"]["fx_erreur_abs_pct"][i15], 4)},
            "figures": [f]}


if __name__ == "__main__":
    print(run())
