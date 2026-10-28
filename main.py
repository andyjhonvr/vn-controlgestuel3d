"""
Contrôle d'un lecteur vidéo par gestes (itérations 1 à 8).
"""
import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from g3d.calibration import focal_for_frame, load_mono, load_stereo
from g3d.config import Config
from g3d.depth import StereoRig, calibrate_palm_radius, load_palm, mono_depth_cm, save_palm
from g3d.gestures import EventLogger, GestureEngine, Sample
from g3d.hand import HandSegmenter, draw_hand, extract_hand, hand_protect_mask
from g3d.player import VideoPlayer
from g3d.zone import ControlZone

ROOT = Path(__file__).resolve().parent
CORNER_NAMES = ["haut-gauche", "haut-droit", "bas-droit", "bas-gauche"]


def diagnostic_mosaic(seg: HandSegmenter, final_mask: np.ndarray) -> np.ndarray:
    """Trois masques côte à côte : peau, avant-plan, résultat final."""
    panels = []
    for name, m in [("Peau (YCrCb)", seg.last_skin), ("Avant-plan", seg.last_fg), ("Final", final_mask)]:
        if m is None:
            m = np.zeros_like(final_mask)
        p = cv2.cvtColor(cv2.resize(cv2.flip(m, 1), (320, 240)), cv2.COLOR_GRAY2BGR)
        cv2.putText(p, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panels.append(p)
    return np.hstack(panels)


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Caméra {index} introuvable")
    return cap


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--cam2", type=int, default=None, help="2e caméra pour la stéréo (itération 8)")
    ap.add_argument("--video", default=None, help="vidéo à piloter (sinon vidéo synthétique)")
    ap.add_argument("--iteration", type=int, choices=range(1, 9), default=8,
                    help="1 : détection; 2 : doigts; 3 : lecture et balayages; 4 : volume; "
                         "5 : gestes robustes; 6 : zone; 7 : étalonnage; 8 : stéréo (défaut, toutes les fonctions)")
    ap.add_argument("--zone", default=str(ROOT / "data" / "zone.json"))
    ap.add_argument("--calib", default=str(ROOT / "data" / "mono.npz"))
    ap.add_argument("--stereo", default=str(ROOT / "data" / "stereo.npz"))
    ap.add_argument("--palm", default=str(ROOT / "data" / "palm.json"))
    ap.add_argument("--log", default=str(ROOT / "results" / "events_log.csv"))
    args = ap.parse_args(argv)
    it = args.iteration
    detection_only = it == 1
    gestures_enabled = it >= 3
    volume_enabled = it >= 4
    robust = it >= 5                 # anti-retour, réarmement complet, tolérance, CSV
    zone_enabled = it >= 6
    calib_enabled = it >= 7

    cfg = Config()
    cap = _open_camera(args.cam)

    # Itération 7 : vraie focale et rayon de paume étalonné
    mono = None
    if calib_enabled:
        mono = load_mono(args.calib)
        if mono is None:
            print(f"Pas d'étalonnage ({args.calib}) : focale provisoire {cfg.depth.focal_px:.0f} px. "
                  "Lancer scripts/capture_calibration.py.")
        else:
            print(f"Étalonnage chargé : fx={mono['K'][0, 0]:.1f} px, RMS={mono['rms']:.3f} px")
        palm = load_palm(args.palm)
        if palm is not None:
            cfg.depth.palm_radius_cm = palm
            print(f"Rayon de paume étalonné : R={palm:.2f} cm")
    radii = deque(maxlen=cfg.depth.palm_calib_frames)

    # Itération 8 : stéréo si une 2e caméra ET data/stereo.npz sont disponibles
    rig, cap2, seg2, prev_point2 = None, None, None, None
    if it >= 8:
        st = load_stereo(args.stereo)
        if args.cam2 is None or st is None:
            print("Stéréo inactive (--cam2 ou data/stereo.npz manquant) : profondeur monoculaire.")
        else:
            cap2 = _open_camera(args.cam2)
            rig = StereoRig.from_calib(st)
            seg2 = HandSegmenter(cfg.seg)
            print(f"Stéréo active : base {rig.baseline_cm:.1f} cm, RMS {st['rms']:.3f} px")

    # Itération 6 : zone de contrôle
    zone = ControlZone.load(args.zone, cfg.zone) if zone_enabled else None
    if zone_enabled:
        print("Zone chargée." if zone is not None else "Aucune zone : toute l'image compte (z pour définir).")
    ui = {"clicks": None}             # liste des clics pendant la définition de la zone

    def on_mouse(event, x, y, flags, param):
        if ui["clicks"] is not None and event == cv2.EVENT_LBUTTONDOWN:
            ui["clicks"].append((x, y))

    logger = EventLogger(args.log) if robust else None
    seg = HandSegmenter(cfg.seg)
    engine = None                                # créé à la 1re image (largeur connue)
    player = (VideoPlayer(args.video, cfg.player, volume=cfg.gestures.volume_initial_pct,
                          show_volume=volume_enabled)
              if gestures_enabled else None)
    last_reason = ""
    show_mask, show_diag = True, False
    prev_point = None
    mouse_ready = False
    print("Gardez la main hors du champ pendant l'apprentissage du fond (~1 s).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        width = frame.shape[1]
        # Fin de la définition de la zone (le 4e clic arrive pendant waitKey)
        if ui["clicks"] is not None and len(ui["clicks"]) == 4:
            zone = ControlZone.from_clicks(ui["clicks"], width, cfg.zone, mirror=True)
            zone.save(args.zone)
            ui["clicks"] = None
            print(f"Zone enregistrée dans {args.zone} : feuille {zone.size_cm[0]:.2f} × "
                  f"{zone.size_cm[1]:.2f} cm")

        if mono is not None and engine is None:
            cfg.depth.focal_px = focal_for_frame(mono["K"], mono["size"], width)
        # Calculs sur l'image brute; le miroir sert seulement à l'affichage
        mask = seg.segment(frame)
        obs = extract_hand(mask, cfg.seg, prev_point, cfg.fingers,
                           analyze_fingers=not detection_only)
        prev_point = obs.point if obs is not None else None
        seg.update_background(hand_protect_mask(mask.shape, obs))
        if obs is not None:
            radii.append(obs.palm_radius)

        obs2, mask2, frame2 = None, None, None
        if cap2 is not None:
            ok2, frame2 = cap2.read()
            if ok2:
                mask2 = seg2.segment(frame2)
                obs2 = extract_hand(mask2, cfg.seg, prev_point2, cfg.fingers, analyze_fingers=False)
                prev_point2 = obs2.point if obs2 is not None else None
                seg2.update_background(hand_protect_mask(mask2.shape, obs2))

        # Main → échantillon → moteur de gestes → lecteur
        z, in_zone = None, True
        if gestures_enabled:
            if engine is None:
                engine = GestureEngine(width, cfg.gestures, volume_enabled=volume_enabled, robust=robust)
            if volume_enabled and obs is not None:
                if rig is not None:
                    z = rig.depth_cm(obs.point, obs2.point) if obs2 is not None else None
                else:
                    z = mono_depth_cm(obs.palm_radius, cfg.depth)
            if zone is not None and obs is not None:
                in_zone = zone.contains(obs.point[0], obs.point[1], z, use_z=rig is not None)
            sample = None
            if obs is not None and not seg.warming_up and ui["clicks"] is None:
                sample = Sample(time.monotonic(), obs.point[0], obs.point[1], obs.fingers, z, in_zone)
            event = engine.update(sample)
            if event is not None:
                player.handle(event)
                last_reason = str(event)
                print(last_reason)
                if logger is not None:
                    logger.log(event)
            while engine.suppressed:
                blocked = engine.suppressed.popleft()
                print(blocked)
                if logger is not None:
                    logger.log(blocked)

        vis = draw_hand(frame, obs) if obs is not None else frame.copy()
        if zone is not None:
            zone.draw(vis, in_zone if obs is not None else None)
        if ui["clicks"] is not None:
            for k, (cx, cy) in enumerate(ui["clicks"]):
                cv2.circle(vis, (width - 1 - cx, cy), 6, (0, 255, 255), -1)
        vis = cv2.flip(vis, 1)
        if seg.warming_up:
            txt = "apprentissage du fond... retirez la main"
        elif obs is not None:
            bras = "oui" if obs.arm_entry is not None else "non"
            if detection_only:
                txt = f"Iteration 1 : main detectee  r={obs.palm_radius:.0f} px  bras={bras}"
            else:
                forme = "POING" if obs.fingers == 0 else ("OUVERTE" if obs.fingers >= 4 else "autre")
                txt = (f"doigts : {obs.fingers} ({forme})  solidite={obs.solidity:.2f}  "
                       f"r={obs.palm_radius:.0f} px  ")
                if volume_enabled:
                    src = " st" if rig is not None else ""
                    txt += f"Z{src}={z:.0f} cm  " if z is not None else f"Z{src}=-- cm  "
                txt += f"bras={bras}"
            if engine is not None and engine.gripping:
                zf = f"{engine.z_filtered:.1f} cm" if engine.z_filtered is not None else "..."
                cv2.putText(vis, f"VOLUME (poing) Z={zf}  vol={engine.volume:.0f} %", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            txt = "aucune main"
        if zone_enabled:
            if ui["clicks"] is not None:
                n = len(ui["clicks"])
                ztxt, zcol = f"ZONE : cliquez le coin {CORNER_NAMES[n]} ({n + 1}/4), Echap = annuler", (0, 255, 255)
            elif zone is None:
                ztxt, zcol = "zone : image entiere (z = definir la feuille)", (200, 200, 200)
            elif obs is None:
                ztxt, zcol = f"zone : feuille {zone.size_cm[0]:.1f} x {zone.size_cm[1]:.1f} cm", (0, 200, 0)
            elif in_zone:
                u, v = zone.to_sheet(*obs.point)
                ztxt, zcol = f"DANS LA ZONE  main sur la feuille : ({u:.1f}, {v:.1f}) cm", (0, 200, 0)
            else:
                ztxt, zcol = "HORS ZONE : gestes ignores", (0, 0, 255)
            if rig is not None and zone is not None:
                ztxt += f"  Z in [{zone.cfg.z_min_cm:.0f}, {zone.cfg.z_max_cm:.0f}] cm"
            cv2.putText(vis, ztxt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zcol, 2)
        if calib_enabled:
            ctxt = (f"f={cfg.depth.focal_px:.0f} px ({'etalonnee' if mono else 'provisoire'})  "
                    f"R={cfg.depth.palm_radius_cm:.2f} cm  (c = etalonner R, poing a "
                    f"{cfg.depth.palm_calib_distance_cm:.0f} cm)")
            cv2.putText(vis, ctxt, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(vis, txt, (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if last_reason:
            cv2.putText(vis, last_reason[:95], (10, vis.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
        cv2.imshow("Camera", vis)
        if zone_enabled and not mouse_ready:
            cv2.setMouseCallback("Camera", on_mouse)
            mouse_ready = True
        if player is not None:
            cv2.imshow("Lecteur", player.render())
        if show_mask:
            cv2.imshow("Masque", cv2.flip(mask, 1))
        if show_diag:
            cv2.imshow("Diagnostic", diagnostic_mosaic(seg, mask))
        if cap2 is not None and frame2 is not None:
            v2 = draw_hand(frame2, obs2) if obs2 is not None else frame2.copy()
            cv2.imshow("Camera 2", cv2.resize(cv2.flip(v2, 1), (320, 240)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and player is not None:
            player.toggle()
        if key == ord("b"):
            seg.reset_background()
            if seg2 is not None:
                seg2.reset_background()
            prev_point = prev_point2 = None
            print("Réapprentissage du fond...")
        if key == ord("d"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Masque")
        if key == ord("m"):
            show_diag = not show_diag
            if not show_diag:
                cv2.destroyWindow("Diagnostic")
        if zone_enabled and key == ord("z"):
            ui["clicks"] = []
            print("Cliquez les 4 coins de la feuille tels que vous les voyez : " + ", ".join(CORNER_NAMES))
        if zone_enabled and key == 27:                   # Échap
            ui["clicks"] = None
        if zone_enabled and key == ord("x"):
            zone = None
            if os.path.exists(args.zone):
                os.remove(args.zone)
            print("Zone effacée : toute l'image compte.")
        if calib_enabled and key == ord("c"):
            if obs is None or len(radii) < radii.maxlen:
                print("Étalonnage de R : tenez le poing immobile dans l'image, puis réessayez.")
            else:
                r_med = float(np.median(radii))
                R = calibrate_palm_radius(r_med, cfg.depth.palm_calib_distance_cm, cfg.depth.focal_px)
                cfg.depth.palm_radius_cm = R
                save_palm(args.palm, R)
                print(f"R étalonné : r={r_med:.1f} px à {cfg.depth.palm_calib_distance_cm:.0f} cm, "
                      f"f={cfg.depth.focal_px:.0f} px → R={R:.2f} cm (enregistré)")

    cap.release()
    if cap2 is not None:
        cap2.release()
    if player is not None:
        player.release()
    if logger is not None:
        logger.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
