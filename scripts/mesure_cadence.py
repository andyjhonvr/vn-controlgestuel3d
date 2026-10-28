"""Mesure la cadence (images/s) de la détection de la main, en mono ou en stéréo.

    python scripts/mesure_cadence.py            # caméra 0
    python scripts/mesure_cadence.py 0 1        # deux caméras (stéréo)
Même traitement que main.py par image (segmentation, choix de la main, mise à jour
du fond, affichage) sur 300 images; les 30 premières (apprentissage du fond) ne
sont pas chronométrées. Une source peut aussi être un fichier vidéo.
"""
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g3d.hand import HandSegmenter, extract_hand, hand_protect_mask  # noqa: E402

N_IMAGES, N_IGNOREES = 300, 30


def main():
    srcs = [int(a) if a.isdigit() else a for a in sys.argv[1:]] or [0]
    caps = [cv2.VideoCapture(s) for s in srcs]
    for c, s in zip(caps, srcs):
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not c.isOpened():
            raise SystemExit(f"Caméra {s} introuvable")
    segs = [HandSegmenter() for _ in caps]
    prev = [None] * len(caps)
    t0 = None
    for n in range(N_IMAGES):
        if n == N_IGNOREES:
            t0 = time.time()
        for i, (c, s) in enumerate(zip(caps, segs)):
            ok, f = c.read()
            if not ok:
                raise SystemExit("Lecture de la caméra impossible")
            m = s.segment(f)
            o = extract_hand(m, s.cfg, prev[i])
            prev[i] = o.point if o is not None else None
            s.update_background(hand_protect_mask(m.shape, o))
            cv2.imshow(f"cam{i}", cv2.flip(f, 1))
        cv2.waitKey(1)
    fps = (N_IMAGES - N_IGNOREES) / (time.time() - t0)
    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    print(f"Cadence : {fps:.1f} images/s avec {len(caps)} caméra(s)")


if __name__ == "__main__":
    main()
