"""Expérience 1 : comptage des doigts et centre de paume sur des mains synthétiques.
"""
import numpy as np

from common import BLEU, ORANGE, plt, save
from g3d.config import SegmentationConfig
from g3d.hand import extract_hand
from test_iter2_doigts import CAS, classe, main_synthetique
from test_iter4_volume import bruit_bords

CENTRE = (320, 250)


def evaluer(difficile: bool, seed: int):
    rng = np.random.default_rng(seed)
    conf = np.zeros((6, 7), int)                       # attendu 0-5 × lu 0-5 (+ « aucune main »)
    err_paume, exact, utiles, n = [], 0, 0, 0
    angles = (-45, -20, 0, 20, 45) if difficile else (-30, -15, 0, 15, 30)
    tailles = (30.0, 40.0, 50.0) if difficile else (38.0, 44.0, 50.0)
    for tendus, att in CAS:
        for gauche in (False, True):
            for ang in angles:
                for R in tailles:
                    m = main_synthetique(tendus, R=R, angle_deg=ang, gauche=gauche, centre=CENTRE,
                                         ecarte=0.55 if difficile else 1.0, rng=rng)
                    if difficile:
                        m = bruit_bords(m, rng)
                    obs = extract_hand(m, SegmentationConfig())
                    lu = obs.fingers if obs is not None else 6
                    conf[att, lu] += 1
                    exact += lu == att
                    utiles += lu != 6 and classe(lu) == classe(att)
                    n += 1
                    if obs is not None:
                        err_paume.append(np.hypot(obs.palm_center[0] - CENTRE[0], obs.palm_center[1] - CENTRE[1]) / R)
    return {"images": n, "exact_pct": 100 * exact / n, "classes_utiles_pct": 100 * utiles / n,
            "paume_erreur_mediane_r": float(np.median(err_paume)), "conf": conf, "err": np.array(err_paume)}


def run():
    nom = evaluer(False, 7)
    dif = evaluer(True, 8)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, res, titre in ((axs[0], nom, "Nominal"), (axs[1], dif, "Difficile")):
        c = res["conf"][:, :6]
        ax.imshow(c, cmap="Blues")
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                if c[i, j]:
                    ax.text(j, i, c[i, j], ha="center", va="center",
                            color="white" if c[i, j] > c.max() / 2 else "black", fontsize=9)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xlabel("doigts lus")
        ax.set_ylabel("doigts attendus")
        ax.set_title(f"{titre} : {res['exact_pct']:.1f} % exact, classes utiles {res['classes_utiles_pct']:.1f} %")
    f1 = save(fig, "fig1_doigts_confusion.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1.0, 26)
    ax.hist(nom["err"], bins, color=BLEU, alpha=0.75, label=f"nominal (médiane {nom['paume_erreur_mediane_r']:.2f} r)")
    ax.hist(dif["err"], bins, color=ORANGE, alpha=0.6, label=f"difficile (médiane {dif['paume_erreur_mediane_r']:.2f} r)")
    ax.set_xlabel("erreur du centre de paume / rayon de paume r")
    ax.set_ylabel("images")
    ax.set_title("Centre de paume (profil de largeur depuis le bout de la main)")
    ax.legend()
    f2 = save(fig, "fig2_paume_erreur.png")
    strip = lambda r: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("conf", "err")}
    return {"nominal": strip(nom), "difficile": strip(dif), "figures": [f1, f2]}


if __name__ == "__main__":
    r = run()
    for k in ("nominal", "difficile"):
        print(k, r[k])
