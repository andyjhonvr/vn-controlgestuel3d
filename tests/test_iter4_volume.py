"""Itération 4 : volume par la profondeur monoculaire.

1. Z mono sur des masques synthétiques de poing, de 30 à 80 cm (rayon r = f·R/Z) :
   biais brut, puis erreur après étalonnage en un point (50 cm).
2. Moteur de volume sur des trajectoires de Z bruitées (bruit typique du mono) :
   poing à 50 cm, approché à 40 cm → attendu 50 % + 2 %/cm × 10 cm = 70 %, moins la zone morte
   de 1 cm (−2 %) → 68 %.
3. Chaîne complète image → main → Z → volume sur des masques de poing qui s'approchent.

Lancer : python3 tests/test_iter4_volume.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g3d.config import DepthConfig, GestureConfig, SegmentationConfig  # noqa: E402
from g3d.depth import mono_depth_cm, radius_px_at  # noqa: E402
from g3d.gestures import GestureEngine, Sample  # noqa: E402
from g3d.hand import extract_hand  # noqa: E402
from test_iter2_doigts import main_synthetique  # noqa: E402

FPS = 30.0
DCFG = DepthConfig()


def bruit_bords(m, rng):
    n = rng.normal(0, 60, m.shape)
    b = cv2.GaussianBlur(m.astype(np.float32) + n, (7, 7), 0)
    return ((b > 127) * 255).astype(np.uint8)


def poing(z_cm, rng, ang=0.0, bruit=True):
    """Masque d'un poing dont le rayon de paume vaut f·R/Z (le générateur a r ≈ R)."""
    m = main_synthetique(set(), R=radius_px_at(z_cm, DCFG), angle_deg=ang, rng=rng)
    return bruit_bords(m, rng) if bruit else m


# --------------------------------------------------------------------- 1. Z mono
def test_profondeur():
    rng = np.random.default_rng(11)
    zs = [30, 40, 50, 60, 70, 80]
    mesures = {z: [] for z in zs}
    for z in zs:
        for ang in (-25, -10, 0, 10, 25):
            for _ in range(4):
                obs = extract_hand(poing(z, rng, ang), SegmentationConfig())
                mesures[z].append(mono_depth_cm(obs.palm_radius, DCFG))
    k = 50.0 / np.median(mesures[50])                    # étalonnage en un point
    print("1. Profondeur mono sur des poings synthétiques (bords bruités, 20 images par distance) :")
    print("     Z vrai | Z brut moyen | erreur après étalonnage à 50 cm (moyenne ± écart type)")
    pires = []
    for z in zs:
        brut = np.array(mesures[z])
        err = k * brut - z
        pires.append(np.abs(err).mean())
        print(f"     {z:4d} cm |   {brut.mean():6.1f} cm  |   {err.mean():+5.2f} ± {err.std():.2f} cm")
    return k, max(pires)


# ------------------------------------------------------------ 2. moteur de volume
def trajectoire_z(rng, z0=50.0, z1=40.0, sigma=1.5, rates=3):
    """Poing tenu 0,6 s à z0, approché en 1 s à z1, tenu 0,6 s. `rates` images mal comptées."""
    zs = np.concatenate([np.full(18, z0), z0 + (z1 - z0) * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 30))),
                         np.full(18, z1)])
    doigts = np.zeros(len(zs), int)
    doigts[rng.choice(np.arange(12, len(zs) - 1), rates, replace=False)] = 2
    return [Sample(i / FPS, 320 + rng.normal(0, 3), 250 + rng.normal(0, 3), int(d), z + rng.normal(0, sigma))
            for i, (z, d) in enumerate(zip(zs, doigts))]


def jouer(samples, cfg=None):
    eng = GestureEngine(640, cfg or GestureConfig())
    evs = [e for e in (eng.update(s) for s in samples) if e is not None]
    return eng, evs


def test_moteur():
    print("2. Moteur de volume (100 essais, poing 50 → 40 cm, attendu 68 %) :")
    res = {}
    for sigma in (0.5, 1.5, 3.0):
        rng = np.random.default_rng(5)
        fin, nev = [], []
        for _ in range(100):
            eng, evs = jouer(trajectoire_z(rng, sigma=sigma))
            fin.append(eng.volume)
            nev.append(len(evs))
        res[sigma] = (np.mean(fin), np.std(fin))
        print(f"     bruit Z σ={sigma:.1f} cm : volume final {np.mean(fin):.1f} ± {np.std(fin):.1f} %, "
              f"{np.mean(nev):.1f} événements")

    # Sans médiane ni hystérésis (avant / après des filtres)
    rng = np.random.default_rng(5)
    brut = GestureConfig(volume_median_n=1, volume_hysteresis_pct=0.0, volume_dead_zone_cm=0.0)
    fin, nev = [], []
    for _ in range(100):
        eng, evs = jouer(trajectoire_z(rng, sigma=1.5), brut)
        fin.append(eng.volume)
        nev.append(len(evs))
    print(f"     sans filtres, σ=1,5 cm : volume final {np.mean(fin):.1f} ± {np.std(fin):.1f} %, "
          f"{np.mean(nev):.1f} événements")

    # Poing immobile bruité : le volume ne doit pas trembler
    for sigma in (0.5, 1.0, 1.5):
        rng = np.random.default_rng(6)
        n = 0
        for _ in range(100):
            s = [Sample(i / FPS, 320, 250, 0, 50 + rng.normal(0, sigma)) for i in range(90)]
            n += len(jouer(s)[1]) > 0
        print(f"     poing immobile 3 s, σ={sigma:.1f} cm : volume modifié dans {n}/100 essais")
        if sigma == 1.0:
            tremble = n

    # Lâcher le poing, le reprendre ailleurs : pas de saut (nouvelle référence Z0)
    s = [Sample(i / FPS, 320, 250, 0, 50.0) for i in range(30)]
    s += [Sample((30 + i) / FPS, 320, 250, 5, 50.0) for i in range(10)]        # main ouverte
    s += [Sample((40 + i) / FPS, 320, 250, 0, 35.0) for i in range(30)]        # poing repris à 35 cm
    eng, evs = jouer(s)
    print(f"     poing repris 15 cm plus près : volume {eng.volume:.0f} % (attendu 50 %, sans saut)")
    return res, tremble, eng.volume


# ------------------------------------------------------- 3. chaîne image → volume
def test_chaine():
    rng = np.random.default_rng(9)
    zs = np.concatenate([np.full(15, 50.0), np.linspace(50, 40, 30), np.full(15, 40.0)])
    eng = GestureEngine(640)
    for i, z in enumerate(zs):
        obs = extract_hand(poing(z, rng), SegmentationConfig())
        zm = mono_depth_cm(obs.palm_radius, DCFG)
        eng.update(Sample(i / FPS, obs.point[0], obs.point[1], obs.fingers, zm))
    print(f"3. Chaîne complète (masques de poing 50 → 40 cm) : volume final {eng.volume:.1f} % (attendu 68 %)")
    return eng.volume


if __name__ == "__main__":
    k, pire = test_profondeur()
    res, tremble, reprise = test_moteur()
    v = test_chaine()
    assert pire < 2.0, "Z mono trop imprécis"
    assert abs(res[1.5][0] - 68) < 3 and res[1.5][1] < 4, "volume biaisé ou instable"
    assert tremble <= 2 and reprise == 50.0
    assert abs(v - 68) < 6, "chaîne complète hors tolérance"
    print("OK")
