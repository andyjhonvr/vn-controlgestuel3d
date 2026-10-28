"""Enregistre des masques de vraies mains étiquetés pour experiments/exp_real_hands.py."""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g3d.config import Config  # noqa: E402
from g3d.hand import HandSegmenter, draw_hand, extract_hand, hand_protect_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / "data" / "real_hands"
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    seg, prev, n_saved = HandSegmenter(cfg.seg), None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = seg.segment(frame)
        obs = extract_hand(mask, cfg.seg, prev, cfg.fingers)
        prev = obs.point if obs is not None else None
        seg.update_background(hand_protect_mask(mask.shape, obs))
        vis = cv2.flip(draw_hand(frame, obs) if obs is not None else frame.copy(), 1)
        lu = "-" if obs is None else str(obs.fingers)
        cv2.putText(vis, f"lu : {lu}   enregistres : {n_saved}   (0-5 = etiquette)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Dataset", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            seg.reset_background()
        if ord("0") <= key <= ord("5") and obs is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{n_saved:03d}"
            label = chr(key)
            cv2.imwrite(str(out / f"{label}_{stamp}.png"), mask)
            cv2.imwrite(str(out / f"{label}_{stamp}_img.png"), frame)
            n_saved += 1
            print(f"enregistré : étiquette {label}, lu {obs.fingers}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
