"""Capture d'images de damier et étalonnage (itérations 7 et 8)"""
import argparse
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g3d.calibration import (calibrate_mono, calibrate_stereo, corners_from_folder,  # noqa: E402
                             find_corners, save_mono, save_stereo)
from g3d.config import CalibrationConfig  # noqa: E402


def etalonner(pts1, pts2, size, cfg, out_dir):
    if len(pts1) < cfg.min_views:
        print(f"Seulement {len(pts1)} vues : il en faut au moins {cfg.min_views}.")
        return
    mono = calibrate_mono(pts1, size, cfg.pattern, cfg.square_cm)
    save_mono(str(out_dir / "mono.npz"), mono)
    K = mono["K"]
    verdict = "OK" if mono["rms"] < cfg.max_rms_px else "TROP ÉLEVÉ (refaire des vues plus nettes)"
    print(f"Mono : {len(pts1)} vues, RMS = {mono['rms']:.3f} px ({verdict})")
    print(f"       fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} px")
    print(f"       pire vue : {max(mono['per_view']):.3f} px → {out_dir / 'mono.npz'}")
    if pts2:
        st = calibrate_stereo(pts1, pts2, size, cfg.pattern, cfg.square_cm)
        save_stereo(str(out_dir / "stereo.npz"), st)
        print(f"Stéréo : RMS = {st['rms']:.3f} px (cam1 {st['rms1']:.3f}, cam2 {st['rms2']:.3f}), "
              f"base = {float((st['T'] ** 2).sum() ** 0.5):.2f} cm → {out_dir / 'stereo.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--cam2", type=int, default=None)
    ap.add_argument("--cols", type=int, default=CalibrationConfig().pattern[0], help="coins intérieurs")
    ap.add_argument("--rows", type=int, default=CalibrationConfig().pattern[1])
    ap.add_argument("--square", type=float, default=CalibrationConfig().square_cm, help="côté d'une case (cm)")
    ap.add_argument("--from-images", action="store_true", help="étalonner avec data/calib_images sans caméra")
    args = ap.parse_args()
    cfg = CalibrationConfig(pattern=(args.cols, args.rows), square_cm=args.square)
    out_dir = ROOT / "data"
    img_dir = out_dir / "calib_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.from_images:
        n1, p1, size = corners_from_folder(str(img_dir), cfg.pattern, "cam1_")
        n2, p2, _ = corners_from_folder(str(img_dir), cfg.pattern, "cam2_")
        if n2:                                             # garder les paires complètes
            common = sorted(set(f[5:] for f in n1) & set(f[5:] for f in n2))
            p1 = [p1[n1.index("cam1_" + c)] for c in common]
            p2 = [p2[n2.index("cam2_" + c)] for c in common]
        etalonner(p1, p2 if n2 else [], size, cfg, out_dir)
        return

    caps = [cv2.VideoCapture(args.cam)] + ([cv2.VideoCapture(args.cam2)] if args.cam2 is not None else [])
    for c in caps:
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not c.isOpened():
            raise SystemExit("Caméra introuvable")
    pts = [[] for _ in caps]
    size = None
    print("espace = garder la vue, c = étalonner, q = quitter")
    while True:
        frames = []
        for c in caps:
            ok, f = c.read()
            if not ok:
                raise SystemExit("Lecture caméra impossible")
            frames.append(f)
        size = (frames[0].shape[1], frames[0].shape[0])
        corners = [find_corners(f, cfg.pattern) for f in frames]
        for i, (f, cr) in enumerate(zip(frames, corners)):
            v = f.copy()
            if cr is not None:
                cv2.drawChessboardCorners(v, cfg.pattern, cr, True)
            col = (0, 255, 0) if cr is not None else (0, 0, 255)
            cv2.putText(v, f"vues : {len(pts[0])}  damier {'trouve' if cr is not None else 'absent'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.imshow(f"Etalonnage cam{i + 1}", v)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and all(c is not None for c in corners):
            n = len(pts[0])
            for i, (f, cr) in enumerate(zip(frames, corners)):
                cv2.imwrite(str(img_dir / f"cam{i + 1}_{n:03d}.png"), f)
                pts[i].append(cr)
            print(f"Vue {n + 1} gardée")
        if key == ord("c"):
            etalonner(pts[0], pts[1] if len(caps) > 1 else [], size, cfg, out_dir)
    for c in caps:
        c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
