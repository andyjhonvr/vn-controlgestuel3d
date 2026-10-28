"""Itération 8 : stéréo 3D (deux caméras étalonnées, triangulation du centre de paume).
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(__file__))
from g3d.calibration import calibrate_stereo, find_corners  # noqa: E402
from g3d.config import DepthConfig, GestureConfig, SegmentationConfig, ZoneConfig  # noqa: E402
from g3d.depth import StereoRig, mono_depth_cm, radius_px_at  # noqa: E402
from g3d.geometry import triangulate_dlt  # noqa: E402
from g3d.gestures import GestureEngine, Sample  # noqa: E402
from g3d.hand import extract_hand  # noqa: E402
from g3d.zone import ControlZone  # noqa: E402
from synth import DIST, camera_K, project, random_board_poses, render_board, stereo_pair  # noqa: E402
from test_iter2_doigts import main_synthetique  # noqa: E402

PATTERN, SQ, B = (9, 6), 2.5, 10.0
K1, K2 = camera_K(600.0), camera_K(610.0, 318.0, 242.0)
R12, T12 = stereo_pair(B)
R_PAUME = 4.0


def test_etalonnage():
    rng = np.random.default_rng(21)
    p1, p2 = [], []
    for rv, tv in random_board_poses(20, rng, PATTERN, SQ, z_range=(40, 65)):
        Rb, _ = cv2.Rodrigues(rv)
        rv2 = cv2.Rodrigues(R12 @ Rb)[0]
        tv2 = R12 @ tv + T12
        c1 = find_corners(render_board(K1, DIST, rv, tv, PATTERN, SQ, rng=rng), PATTERN)
        c2 = find_corners(render_board(K2, DIST, rv2, tv2, PATTERN, SQ, rng=rng), PATTERN)
        if c1 is not None and c2 is not None:
            p1.append(c1)
            p2.append(c2)
    st = calibrate_stereo(p1, p2, (640, 480), PATTERN, SQ)
    base = np.linalg.norm(st["T"])
    dR = np.degrees(np.linalg.norm(cv2.Rodrigues(st["R"] @ R12.T)[0]))
    print(f"1. Étalonnage stéréo : {len(p1)} paires, RMS = {st['rms']:.3f} px "
          f"(cam1 {st['rms1']:.3f}, cam2 {st['rms2']:.3f}), base {base:.3f} cm (vraie {B:.1f}), "
          f"erreur de rotation {dR:.3f}°")
    return st


def rig_vrai():
    return StereoRig(K1, np.zeros(5), K2, np.zeros(5), R12, T12)


def test_dlt():
    rig = rig_vrai()
    rng = np.random.default_rng(1)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R12, T12.reshape(3, 1)])
    ecart = ecart_rig = 0.0
    for _ in range(100):
        X = np.array([rng.uniform(-15, 15), rng.uniform(-10, 10), rng.uniform(30, 80)])
        x1 = project(K1, np.eye(3), np.zeros(3), X) + rng.normal(0, 1, 2)
        x2 = project(K2, R12, T12, X) + rng.normal(0, 1, 2)
        a = triangulate_dlt(P1, P2, x1, x2)
        h = cv2.triangulatePoints(P1, P2, x1.reshape(2, 1), x2.reshape(2, 1)).ravel()
        b = h[:3] / h[3]
        ecart = max(ecart, np.abs(a - b).max())
        ecart_rig = max(ecart_rig, np.abs(rig.triangulate(x1, x2) - b).max())
    print(f"2. Triangulation DLT maison vs cv2.triangulatePoints (pixels) : écart max {ecart:.1e} cm; "
          f"StereoRig (coordonnées normalisées) : {ecart_rig:.1e} cm")
    return ecart, ecart_rig


def erreurs_z(sigma_px=1.0, n=500, seed=5):
    """Erreur absolue moyenne sur Z (cm) stéréo et mono pour chaque distance."""
    rng = np.random.default_rng(seed)
    rig, dcfg = rig_vrai(), DepthConfig(focal_px=600.0, palm_radius_cm=R_PAUME)
    out = {}
    for z in (30, 40, 50, 60, 70, 80):
        es, em = [], []
        for _ in range(n):
            X = np.array([rng.uniform(-8, 8), rng.uniform(-6, 6), z])
            x1 = project(K1, np.eye(3), np.zeros(3), X) + rng.normal(0, sigma_px, 2)
            x2 = project(K2, R12, T12, X) + rng.normal(0, sigma_px, 2)
            es.append(rig.depth_cm(x1, x2) - z)
            em.append(mono_depth_cm(radius_px_at(z, dcfg) + rng.normal(0, sigma_px), dcfg) - z)
        out[z] = (np.abs(es).mean(), np.abs(em).mean(), np.std(es), np.std(em))
    return out


def test_bruit():
    res = erreurs_z()
    print("3. Erreur sur Z (bruit de 1 px), moyenne des |erreurs| :")
    print("     Z vrai |  stéréo  |   mono")
    for z, (s, m, _, _) in res.items():
        print(f"     {z:4d} cm | {s:5.2f} cm | {m:5.2f} cm")
    return res


def test_chaine():
    """Masques de main dans les deux images (même main 3D), Z triangulé contre Z vrai."""
    rng = np.random.default_rng(8)
    rig = rig_vrai()
    errs = []
    for z in (35, 45, 55, 65, 75):
        n = 0
        while n < 6:
            X = np.array([rng.uniform(-6, 6), rng.uniform(-3, 5), z])
            # la main (≈ 2,5 r autour de la paume) doit être entière dans les deux images
            vis = all(2.5 * K[0, 0] * R_PAUME / z < project(K, R, T, X)[0] < 640 - 2.5 * K[0, 0] * R_PAUME / z
                      for K, R, T in ((K1, np.eye(3), np.zeros(3)), (K2, R12, T12)))
            if not vis:
                continue
            n += 1
            pts = []
            for K, R, T in ((K1, np.eye(3), np.zeros(3)), (K2, R12, T12)):
                c = project(K, R, T, X)
                r = K[0, 0] * R_PAUME / (R @ X + T)[2]
                m = main_synthetique({"index", "majeur"}, R=r, centre=(int(round(c[0])), int(round(c[1]))), rng=rng)
                obs = extract_hand(m, SegmentationConfig())
                pts.append(obs.point if obs is not None else None)
            zt = rig.depth_cm(*pts)
            errs.append(abs(zt - z) if zt is not None else np.inf)
    print(f"4. Chaîne image → Z stéréo (30 mains entières dans les 2 images, 35 → 75 cm) : "
          f"erreur moyenne {np.mean(errs):.2f} cm, "
          f"max {np.max(errs):.2f} cm")
    return float(np.mean(errs))


def test_zone3d():
    corners = np.array([[180, 150], [460, 150], [500, 380], [140, 380]], float)
    z = ControlZone(corners, (27.94, 21.59), ZoneConfig(z_min_cm=30, z_max_cm=70))
    c = (320, 260)
    res = [z.contains(*c, 50, use_z=True), z.contains(*c, 20, use_z=True), z.contains(*c, 85, use_z=True),
           z.contains(*c, None, use_z=True), z.contains(600, 260, 50, use_z=True), z.contains(*c, 20)]
    attendu = [True, False, False, False, False, True]
    print(f"5. Zone 3D (feuille + Z ∈ [30, 70] cm) : {sum(a == b for a, b in zip(res, attendu))}/6 cas corrects")
    return res == attendu


def test_volume(res):
    """Bruit de Z mesuré au §3 à 50 cm, appliqué au moteur de volume."""
    out = {}
    for nom, sigma in (("stéréo", res[50][2]), ("mono", res[50][3])):
        rng = np.random.default_rng(5)
        fins = []
        for _ in range(100):
            zs = np.concatenate([np.full(18, 50.0), 50 - 10 * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 30))),
                                 np.full(18, 40.0)])
            eng = GestureEngine(640, GestureConfig())
            for i, z in enumerate(zs):
                eng.update(Sample(i / 30, 320, 250, 0, z + rng.normal(0, sigma)))
            fins.append(eng.volume)
        out[nom] = (np.mean(fins), np.std(fins), sigma)
    print("6. Volume 50 → 40 cm (attendu 68 %), bruit de Z à 50 cm :")
    for nom, (m, s, sig) in out.items():
        print(f"     {nom:<6} (σ_Z = {sig:.2f} cm) : {m:.1f} ± {s:.1f} %")
    return out


if __name__ == "__main__":
    st = test_etalonnage()
    e = test_dlt()
    res = test_bruit()
    ch = test_chaine()
    z3 = test_zone3d()
    vol = test_volume(res)
    assert st["rms"] < 0.5 and abs(np.linalg.norm(st["T"]) - B) < 0.1
    assert e[0] < 1e-6 and e[1] < 0.05
    assert all(res[z][0] < res[z][1] for z in res), "la stéréo doit battre le mono"
    assert ch < 1.5 and z3
    assert abs(vol["stéréo"][0] - 68) < 2 and vol["stéréo"][1] <= vol["mono"][1]
    print("OK")
