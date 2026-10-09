"""
Contrôle d'un lecteur vidéo par gestes (itérations 1 à 4).

Au démarrage, gardez la main hors du champ pendant ~1 s (apprentissage du fond).
Gestes : balayage gauche / droite = ∓10 s; main ouverte immobile 0,8 s = lecture / pause;
         poing fermé puis approcher / éloigner = volume + / −.
Touches : b = réapprendre le fond (main HORS du champ), d = afficher/masquer le masque,
          m = diagnostic (peau, avant-plan, final), espace = lecture / pause, q = quitter.
Options : --cam N (caméra), --video fichier.mp4 (sinon vidéo synthétique).
"""
import argparse
import time

import cv2
import numpy as np

from g3d.config import Config
from g3d.depth import mono_depth_cm
from g3d.gestures import GestureEngine, Sample
from g3d.hand import HandSegmenter, draw_hand, extract_hand, hand_protect_mask
from g3d.player import VideoPlayer


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", default=None, help="vidéo à piloter (sinon vidéo synthétique)")
    args = ap.parse_args()

    cfg = Config()
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Caméra {args.cam} introuvable")

    seg = HandSegmenter(cfg.seg)
    engine = None                                # créé à la 1re image (largeur connue)
    player = VideoPlayer(args.video, cfg.player, volume=cfg.gestures.volume_initial_pct)
    last_reason = ""
    show_mask, show_diag = True, False
    prev_point = None
    print("Gardez la main hors du champ pendant l'apprentissage du fond (~1 s).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Calculs sur l'image brute; le miroir sert seulement à l'affichage
        mask = seg.segment(frame)
        obs = extract_hand(mask, cfg.seg, prev_point, cfg.fingers)
        prev_point = obs.point if obs is not None else None
        seg.update_background(hand_protect_mask(mask.shape, obs))

        # Main → échantillon → moteur de gestes → lecteur
        if engine is None:
            engine = GestureEngine(frame.shape[1], cfg.gestures)
        z = mono_depth_cm(obs.palm_radius, cfg.depth) if obs is not None else None
        sample = None
        if obs is not None and not seg.warming_up:
            sample = Sample(time.monotonic(), obs.point[0], obs.point[1], obs.fingers, z)
        event = engine.update(sample)
        if event is not None:
            player.handle(event)
            last_reason = str(event)
            print(last_reason)

        vis = draw_hand(frame, obs) if obs is not None else frame.copy()
        vis = cv2.flip(vis, 1)
        if seg.warming_up:
            txt = "apprentissage du fond... retirez la main"
        elif obs is not None:
            bras = "oui" if obs.arm_entry is not None else "non"
            forme = "POING" if obs.fingers == 0 else ("OUVERTE" if obs.fingers >= 4 else "autre")
            txt = (f"doigts : {obs.fingers} ({forme})  solidite={obs.solidity:.2f}  "
                   f"r={obs.palm_radius:.0f} px  Z={z:.0f} cm  bras={bras}")
            if engine.gripping:
                zf = f"{engine.z_filtered:.1f} cm" if engine.z_filtered is not None else "..."
                cv2.putText(vis, f"VOLUME (poing) Z={zf}  vol={engine.volume:.0f} %", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            txt = "aucune main"
        cv2.putText(vis, txt, (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if last_reason:
            cv2.putText(vis, last_reason[:95], (10, vis.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
        cv2.imshow("Camera", vis)
        cv2.imshow("Lecteur", player.render())
        if show_mask:
            cv2.imshow("Masque", cv2.flip(mask, 1))
        if show_diag:
            cv2.imshow("Diagnostic", diagnostic_mosaic(seg, mask))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            player.toggle()
        if key == ord("b"):
            seg.reset_background()
            prev_point = None
            print("Réapprentissage du fond...")
        if key == ord("d"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Masque")
        if key == ord("m"):
            show_diag = not show_diag
            if not show_diag:
                cv2.destroyWindow("Diagnostic")

    cap.release()
    player.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()