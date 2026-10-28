"""Expérience 4 : profondeur stéréo (triangulation) contre profondeur mono (rayon de paume).
"""
import numpy as np

from common import BLEU, ORANGE, plt, save
from test_iter8_stereo import B, K1, R_PAUME, erreurs_z


def run():
    res = erreurs_z(sigma_px=1.0, n=500, seed=5)
    zs = np.array(sorted(res))
    st = np.array([res[z][0] for z in zs])
    mo = np.array([res[z][1] for z in zs])
    st_sd = np.array([res[z][2] for z in zs])
    mo_sd = np.array([res[z][3] for z in zs])
    f = K1[0, 0]
    zz = np.linspace(30, 80, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(zs, st_sd, "o-", color=BLEU, label="stéréo (mesuré, écart type)")
    ax.plot(zs, mo_sd, "s-", color=ORANGE, label="mono (mesuré, écart type)")
    ax.plot(zz, zz ** 2 / (f * B) * np.sqrt(2), ":", color=BLEU, label="stéréo, théorie Z²√2/(fB)")
    ax.plot(zz, zz ** 2 / (f * R_PAUME), ":", color=ORANGE, label="mono, théorie Z²/(fR)")
    ax.set_xlabel("distance réelle Z (cm)")
    ax.set_ylabel("écart type de Z (cm)")
    ax.set_title("Profondeur : stéréo (B = 10 cm) contre mono (R = 4 cm), bruit 1 px")
    ax.legend(fontsize=8)
    fig_path = save(fig, "fig5_triangulation.png")
    return {"z_cm": zs.tolist(),
            "stereo_erreur_moy_cm": np.round(st, 3).tolist(), "mono_erreur_moy_cm": np.round(mo, 3).tolist(),
            "stereo_sigma_cm": np.round(st_sd, 3).tolist(), "mono_sigma_cm": np.round(mo_sd, 3).tolist(),
            "sigma_z_50cm": {"stereo": float(res[50][2]), "mono": float(res[50][3])},
            "figures": [fig_path]}


if __name__ == "__main__":
    print(run())
