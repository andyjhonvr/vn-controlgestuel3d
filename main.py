"""Itération 1 — « Je vois ma main ».

    python main.py            # webcam 0
    python main.py --cam 1

Au démarrage, gardez la main hors du champ pendant ~1 s (apprentissage du fond).
Touches : b = réapprendre le fond (main HORS du champ), d = afficher/masquer le masque,
          m = diagnostic (peau, avant-plan, final), q = quitter.
"""
import argparse

import cv2
import numpy as np

from g3d.config import Config
from g3d.hand import HandSegmenter, draw_hand, extract_hand, hand_protect_mask


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
    args = ap.parse_args()

    cfg = Config()
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Caméra {args.cam} introuvable")

    seg = HandSegmenter(cfg.seg)
    show_mask, show_diag = True, False
    prev_point = None
    print("Gardez la main hors du champ pendant l'apprentissage du fond (~1 s).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Calculs sur l'image brute; le miroir sert seulement à l'affichage
        mask = seg.segment(frame)
        obs = extract_hand(mask, cfg.seg, prev_point)
        prev_point = obs.point if obs is not None else None
        seg.update_background(hand_protect_mask(mask.shape, obs))

        vis = draw_hand(frame, obs) if obs is not None else frame.copy()
        vis = cv2.flip(vis, 1)
        if seg.warming_up:
            txt = "apprentissage du fond... retirez la main"
        elif obs is not None:
            bras = "oui" if obs.arm_entry is not None else "non"
            txt = f"main : aire={obs.area:.0f} px2  rayon paume={obs.palm_radius:.0f} px  bras={bras}"
        else:
            txt = "aucune main"
        cv2.putText(vis, txt, (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("Camera", vis)
        if show_mask:
            cv2.imshow("Masque", cv2.flip(mask, 1))
        if show_diag:
            cv2.imshow("Diagnostic", diagnostic_mosaic(seg, mask))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()