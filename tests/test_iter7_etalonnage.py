"""Itération 7 : étalonnage par damier et Z monoculaire avec la vraie focale.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(__file__))
from g3d.calibration import calibrate_mono, find_corners, focal_for_frame, load_mono, save_mono  # noqa: E402
from g3d.config import DepthConfig, SegmentationConfig  # noqa: E402
from g3d.depth import calibrate_palm_radius, load_palm, mono_depth_cm, radius_px_at, save_palm  # noqa: E402
from g3d.gestures import GestureEngine, Sample  # noqa: E402
from g3d.hand import extract_hand  # noqa: E402
from synth import DIST, camera_K, random_board_poses, render_board  # noqa: E402
from test_iter2_doigts import main_synthetique  # noqa: E402
from test_iter4_volume import bruit_bords  # noqa: E402

F_VRAIE, R_VRAI = 600.0, 4.0
PATTERN, SQ = (9, 6), 2.5


def etalonner(noise, blur, seed=3, n=15):
    rng = np.random.default_rng(seed)
    K = camera_K(F_VRAIE)
    pts = []
    for rv, tv in random_board_poses(n, rng, PATTERN, SQ):
        c = find_corners(render_board(K, DIST, rv, tv, PATTERN, SQ, noise=noise, blur=blur, rng=rng), PATTERN)
        if c is not None:
            pts.append(c)
    r = calibrate_mono(pts, (640, 480), PATTERN, SQ)
    Ke = r["K"]
    return r, len(pts), {"fx": 100 * (Ke[0, 0] / K[0, 0] - 1), "fy": 100 * (Ke[1, 1] / K[1, 1] - 1),
                         "cx": Ke[0, 2] - K[0, 2], "cy": Ke[1, 2] - K[1, 2]}


def test_etalonnage():
    print("1. Étalonnage sur damiers synthétiques (f vraie = 600 px, distorsion k1 = -0,12) :")
    out = {}
    for nom, noise, blur in (("nominal", 2.0, 0.8), ("difficile", 8.0, 1.6)):
        r, n, e = etalonner(noise, blur)
        out[nom] = r
        print(f"     {nom:<9} : {n} vues, RMS = {r['rms']:.3f} px, fx {e['fx']:+.3f} %, fy {e['fy']:+.3f} %, "
              f"cx {e['cx']:+.2f} px, cy {e['cy']:+.2f} px, k1 = {r['dist'][0, 0]:+.3f}")
    return out


def poing(z, rng, f=F_VRAIE):
    m = main_synthetique(set(), R=radius_px_at(z, DepthConfig(focal_px=f, palm_radius_cm=R_VRAI)),
                         angle_deg=rng.uniform(-20, 20), rng=rng)
    return bruit_bords(m, rng)


def test_profondeur(calib):
    rng = np.random.default_rng(11)
    f_cal = focal_for_frame(calib["K"], calib["size"], 640)
    # étalonnage de R : poing tenu à 50 cm, médiane du rayon sur 15 images (touche c)
    r50 = np.median([extract_hand(poing(50, rng), SegmentationConfig()).palm_radius for _ in range(15)])
    R_cal = calibrate_palm_radius(r50, 50.0, f_cal)
    avant, apres = DepthConfig(), DepthConfig(focal_px=f_cal, palm_radius_cm=R_cal)
    print(f"2. Z mono (poings vus par la caméra f = 600 px) : f étalonnée = {f_cal:.1f} px, "
          f"R étalonné à 50 cm = {R_cal:.2f} cm")
    print("     Z vrai | avant (f=500, R=4) | après (f, R étalonnés)")
    ea, ep = [], []
    for z in (30, 40, 50, 60, 70, 80):
        rs = [extract_hand(poing(z, rng), SegmentationConfig()).palm_radius for _ in range(15)]
        za = np.array([mono_depth_cm(r, avant) for r in rs])
        zp = np.array([mono_depth_cm(r, apres) for r in rs])
        ea.append(np.abs(za - z).mean())
        ep.append(np.abs(zp - z).mean())
        print(f"     {z:4d} cm |  {za.mean():5.1f} cm ({za.mean() - z:+5.1f}) | "
              f" {zp.mean():5.1f} cm ({zp.mean() - z:+5.2f} ± {zp.std():.2f})")
    print(f"     erreur absolue moyenne : {np.mean(ea):.2f} cm → {np.mean(ep):.2f} cm")
    return avant, apres, np.mean(ea), np.mean(ep)


def chaine(dcfg):
    rng = np.random.default_rng(9)
    zs = np.concatenate([np.full(15, 50.0), np.linspace(50, 40, 30), np.full(15, 40.0)])
    eng = GestureEngine(640)
    for i, z in enumerate(zs):
        obs = extract_hand(poing(z, rng), SegmentationConfig())
        eng.update(Sample(i / 30.0, obs.point[0], obs.point[1], obs.fingers, mono_depth_cm(obs.palm_radius, dcfg)))
    return eng.volume


def test_fichiers(calib):
    with tempfile.TemporaryDirectory() as d:
        save_mono(os.path.join(d, "mono.npz"), calib)
        save_palm(os.path.join(d, "palm.json"), 4.2)
        m = load_mono(os.path.join(d, "mono.npz"))
        ok = np.allclose(m["K"], calib["K"]) and m["size"] == (640, 480) and load_palm(os.path.join(d, "palm.json")) == 4.2
    print(f"4. mono.npz et palm.json relus à l'identique : {ok}")
    return ok


if __name__ == "__main__":
    res = test_etalonnage()
    avant, apres, ea, ep = test_profondeur(res["nominal"])
    va, vp = chaine(avant), chaine(apres)
    print(f"3. Chaîne image → volume (50 → 40 cm, attendu 68 %) : avant {va:.1f} % → après {vp:.1f} %")
    ok = test_fichiers(res["nominal"])
    assert res["nominal"]["rms"] < 0.5 and res["difficile"]["rms"] < 0.5, "RMS trop élevé"
    assert abs(res["nominal"]["K"][0, 0] / F_VRAIE - 1) < 0.005
    assert ep < 1.0 and ep < ea, "Z mono étalonné imprécis"
    assert abs(vp - 68) < 3 and ok
    print("OK")
