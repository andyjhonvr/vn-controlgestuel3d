"""Itération 6 : zone de contrôle (feuille lettre cliquée, homographie DLT).
"""
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(__file__))
from g3d.config import GestureConfig, ZoneConfig  # noqa: E402
from g3d.geometry import apply_homography, dlt_homography  # noqa: E402
from g3d.gestures import GestureEngine  # noqa: E402
from g3d.zone import ControlZone  # noqa: E402
from synth import letter_sheet_homography  # noqa: E402
from test_iter3_gestes import trajectoire  # noqa: E402

W = 640


def test_dlt():
    rng = np.random.default_rng(1)
    res = {}
    for sigma in (0.0, 1.0):
        ecarts = []
        for _ in range(200):
            sheet, img, _ = letter_sheet_homography(rng)
            extra = rng.uniform([0, 0], [27.94, 21.59], (6, 2))       # 10 points (moindres carrés)
            H_true = cv2.getPerspectiveTransform(sheet.astype(np.float32), img.astype(np.float32))
            src = np.vstack([img, apply_homography(H_true, extra)]) + rng.normal(0, sigma, (10, 2))
            dst = np.vstack([sheet, extra])
            Hm = dlt_homography(src, dst)
            Hc, _ = cv2.findHomography(src, dst, 0)
            test_pts = apply_homography(H_true, rng.uniform([0, 0], [27.94, 21.59], (50, 2)))
            ecarts.append(np.abs(apply_homography(Hm, test_pts) - apply_homography(Hc, test_pts)).max())
        res[sigma] = max(ecarts) * 10                                   # mm
    print(f"1. DLT maison vs cv2.findHomography (200 feuilles, 10 points) : écart max "
          f"{res[0.0]:.1e} mm sans bruit, {res[1.0]:.2f} mm avec σ = 1 px "
          f"(OpenCV raffine ensuite par Levenberg-Marquardt)")
    return res


def clics_ecran(img_pts):
    """Coins de la feuille (image brute) → clics tels que vus à l'écran (miroir), HG, HD, BD, BG."""
    scr = img_pts.copy()
    scr[:, 0] = W - 1 - scr[:, 0]
    # À l'écran, la gauche et la droite sont inversées : HG écran = HD brut
    return scr[[1, 0, 3, 2]]


def test_clics_et_bruit():
    rng = np.random.default_rng(2)
    orient_ok, err0 = 0, []
    res = {}
    for sigma in (0.0, 1.0, 3.0):
        errs = []
        for k in range(200):
            landscape = k % 2 == 0
            sheet, img, (w, h) = letter_sheet_homography(rng, landscape)
            clicks = clics_ecran(img) + rng.normal(0, sigma, (4, 2))
            cfg = ZoneConfig(orientation="paysage" if landscape else "portrait")
            z = ControlZone.from_clicks(clicks, W, cfg)
            if sigma == 0:
                auto = ControlZone.from_clicks(clicks, W, ZoneConfig(orientation="auto"))
                orient_ok += (auto.size_cm[0] > auto.size_cm[1]) == (w > h)
            # vérité : point de la feuille (cm, repère écran) → image brute
            H_true = cv2.getPerspectiveTransform(sheet.astype(np.float32), img.astype(np.float32))
            uv = rng.uniform([0, 0], [w, h], (30, 2))
            px = apply_homography(H_true, uv)
            est = np.array([z.to_sheet(*p) for p in px])
            uv_screen = np.c_[w - uv[:, 0], uv[:, 1]]                 # miroir : u écran = w − u brut
            errs.append(np.linalg.norm(est - uv_screen, axis=1).mean())
        res[sigma] = float(np.mean(errs))
    print(f"2. Orientation \"auto\" déduite des clics (info) : {orient_ok}/200; "
          f"le défaut est \"paysage\" (config)")
    print("3. Erreur de position sur la feuille (cm) selon le bruit des clics :")
    for s, e in res.items():
        print(f"     σ = {s:.0f} px → {e:.3f} cm")
    return orient_ok, res


def test_contains():
    rng = np.random.default_rng(3)
    ok = n = 0
    for _ in range(100):
        sheet, img, (w, h) = letter_sheet_homography(rng)
        z = ControlZone.from_clicks(clics_ecran(img), W)
        H_true = cv2.getPerspectiveTransform(sheet.astype(np.float32), img.astype(np.float32))
        dedans = apply_homography(H_true, rng.uniform([1, 1], [w - 1, h - 1], (20, 2)))
        dehors = apply_homography(H_true, np.r_[rng.uniform([-15, 1], [-1, h - 1], (10, 2)),
                                                rng.uniform([w + 1, 1], [w + 15, h - 1], (10, 2))])
        ok += sum(z.contains(*p) for p in dedans) + sum(not z.contains(*p) for p in dehors)
        n += 40
    print(f"4. contains() : {ok}/{n} points bien classés")
    return ok / n


def test_moteur():
    """Gestes complets hors zone → rien; balayage + sortie de zone + retour → 2 balayages."""
    rng = np.random.default_rng(4)
    hors = ok_retour = 0
    for _ in range(100):
        p = (320, 250)
        s = trajectoire([(0.3, (470, 250), (470, 250), 2), (0.3, (470, 250), (180, 250), 2),
                         (0.3, (180, 250), (180, 250), 2), (2.0, p, p, 5)], rng)
        for x in s:
            x.in_zone = False
        eng = GestureEngine(W, GestureConfig())
        hors += sum(eng.update(x) is not None for x in s) == 0
        # balayage droite, la main sort de la zone 0,2 s (6 images hors zone), revient, balaie à gauche
        s = trajectoire([(0.3, (470, 250), (470, 250), 2), (0.3, (470, 250), (180, 250), 2),
                         (0.2, (180, 250), (180, 250), 2), (0.3, (180, 250), (180, 250), 2),
                         (0.3, (180, 250), (470, 250), 2), (0.3, (470, 250), (470, 250), 2)], rng)
        for x in s[18:24]:
            x.in_zone = False
        eng = GestureEngine(W, GestureConfig())
        evs = [e for e in (eng.update(x) for x in s) if e is not None]
        ok_retour += [(e.kind, e.value) for e in evs] == [("seek", 10.0), ("seek", -10.0)]
    print(f"5. Gestes hors zone sans commande : {hors}/100; sortie de zone puis retour → 2 balayages : "
          f"{ok_retour}/100")
    return hors, ok_retour


def test_json():
    rng = np.random.default_rng(5)
    _, img, _ = letter_sheet_homography(rng)
    z = ControlZone.from_clicks(clics_ecran(img), W, ZoneConfig(z_min_cm=30, z_max_cm=70))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data", "zone.json")
        z.save(path)
        z2 = ControlZone.load(path)
    same = np.allclose(z.H, z2.H) and z2.cfg.z_min_cm == 30 and z2.cfg.z_max_cm == 70
    print(f"6. zone.json relu à l'identique : {same}")
    return same


if __name__ == "__main__":
    dlt = test_dlt()
    orient, res = test_clics_et_bruit()
    c = test_contains()
    hors, retour = test_moteur()
    js = test_json()
    assert dlt[0.0] < 1e-3 and dlt[1.0] < 1.0, "DLT différente d'OpenCV"
    assert res[0.0] < 1e-6 and res[1.0] < 0.3 and res[3.0] < 0.8
    assert c >= 0.99 and hors == 100 and retour == 100 and js
    print("OK")
