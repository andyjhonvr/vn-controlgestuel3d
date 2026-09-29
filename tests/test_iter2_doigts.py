"""Itération 2 : comptage des doigts sur des mains synthétiques.

Chaque main a une paume, un avant-bras qui sort par le bas de l'image, des doigts
tendus et des doigts REPLIÉS (petites bosses des jointures, pouce replié contre la
paume). Les doigts repliés sont le cas qui trompait la règle « n creux → n + 1 ».

Lancer : python3 tests/test_iter2_doigts.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g3d.config import FingerConfig, SegmentationConfig  # noqa: E402
from g3d.hand import extract_hand  # noqa: E402

W, H = 640, 480
DOIGTS = ["pouce", "index", "majeur", "annulaire", "auriculaire"]


def main_synthetique(tendus, R=45.0, angle_deg=0.0, gauche=False, centre=(320, 250),
                     ecarte=1.0, rng=None):
    """Masque binaire d'une main. `tendus` : ensemble de noms de DOIGTS tendus."""
    rng = rng or np.random.default_rng(0)
    m = np.zeros((H, W), np.uint8)
    th = np.radians(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sgn = -1.0 if gauche else 1.0

    def P(x, y):  # coordonnées locales (en R, y vers le haut) → pixels
        v = rot @ np.array([sgn * x * R, -y * R])
        return int(round(centre[0] + v[0])), int(round(centre[1] + v[1]))

    def segment(x0, y0, dir_deg, longueur, largeur):
        a = np.radians(dir_deg)
        x1, y1 = x0 + longueur * np.sin(a), y0 + longueur * np.cos(a)
        cv2.line(m, P(x0, y0), P(x1, y1), 255, max(1, int(largeur * R)))

    # Paume (polygone arrondi) et avant-bras jusqu'au bas de l'image
    paume = [P(x, y) for x, y in [(-1.0, 0.95), (1.0, 0.95), (1.05, -0.6), (0.7, -1.2),
                                  (-0.7, -1.2), (-1.05, -0.6)]]
    cv2.fillPoly(m, [np.array(paume)], 255)
    cv2.circle(m, P(0, 0), int(1.02 * R), 255, -1)
    bas = P(0, -1.0)
    v = rot @ np.array([0, 1.0])
    loin = (int(bas[0] + v[0] * 600), int(bas[1] + v[1] * 600))
    cv2.line(m, bas, loin, 255, int(1.55 * R))

    # Doigts : (x base, y base, direction, longueur tendue, largeur)
    geo = {"index": (-0.72, 0.9, -12 * ecarte, 1.55, 0.40),
           "majeur": (-0.24, 0.95, -4 * ecarte, 1.75, 0.42),
           "annulaire": (0.24, 0.92, 4 * ecarte, 1.60, 0.40),
           "auriculaire": (0.70, 0.80, 14 * ecarte, 1.25, 0.34)}
    for nom, (x0, y0, d, L, larg) in geo.items():
        if nom in tendus:
            segment(x0, y0, d + rng.normal(0, 3), L * rng.uniform(0.92, 1.05), larg)
        else:
            segment(x0, y0 - 0.1, d, 0.28, larg)           # jointure repliée
    if "pouce" in tendus:
        segment(-0.95, 0.0, -55 * ecarte + rng.normal(0, 4), 1.35 * rng.uniform(0.9, 1.05), 0.45)
    else:
        segment(-1.0, -0.3, 35, 0.8, 0.45)                  # pouce replié contre la paume
    m[m > 0] = 255
    return m


CAS = [  # (doigts tendus, nombre attendu)
    (set(), 0),
    ({"index"}, 1),
    ({"index", "majeur"}, 2),
    ({"pouce", "index"}, 2),
    ({"index", "majeur", "annulaire"}, 3),
    ({"pouce", "index", "majeur"}, 3),
    ({"index", "majeur", "annulaire", "auriculaire"}, 4),
    (set(DOIGTS), 5),
]


def classe(n):
    return "poing" if n == 0 else ("ouverte" if n >= 4 else "autre")


def evaluer(cfg=None, verbeux=True):
    rng = np.random.default_rng(7)
    total = exact = cls = 0
    erreurs = {}
    for tendus, attendu in CAS:
        for gauche in (False, True):
            for ang in (-30, -15, 0, 15, 30):
                for R in (38.0, 48.0):
                    mask = main_synthetique(tendus, R=R, angle_deg=ang, gauche=gauche, rng=rng)
                    obs = extract_hand(mask, SegmentationConfig(), None, cfg)
                    n = obs.fingers if obs is not None else -1
                    total += 1
                    exact += n == attendu
                    cls += classe(n) == classe(attendu)
                    if n != attendu:
                        erreurs.setdefault(attendu, []).append(n)
    if verbeux:
        print(f"  exact : {exact}/{total} = {100 * exact / total:.1f} %   "
              f"classes utiles : {100 * cls / total:.1f} %")
        for att, lus in sorted(erreurs.items()):
            print(f"    attendu {att} → lu {sorted(set(lus))} ({len(lus)} erreurs)")
    return exact / total, cls / total


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--images":
        os.makedirs("results", exist_ok=True)
        for i, (tendus, att) in enumerate(CAS):
            cv2.imwrite(f"results/synth_main_{i}_{att}.png", main_synthetique(tendus))
    print("Comptage des doigts (mains synthétiques avec doigts repliés) :")
    ex, cl = evaluer()
    assert ex >= 0.95 and cl == 1.0, "comptage insuffisant"
    print("OK")
