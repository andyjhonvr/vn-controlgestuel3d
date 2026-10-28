"""main.py avec une caméra SIMULÉE, pour les modes --iteration 1 à 8 (sans webcam).
"""
import csv
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(__file__))
import main as app  # noqa: E402
from g3d.calibration import save_stereo  # noqa: E402
from synth import camera_K, stereo_pair  # noqa: E402
from test_iter2_doigts import main_synthetique  # noqa: E402

SKIN = cv2.cvtColor(np.uint8([[[170, 150, 105]]]), cv2.COLOR_YCrCb2BGR)[0, 0]
BG = np.full((480, 640, 3), 40, np.uint8)
DOIGTS5 = {"pouce", "index", "majeur", "annulaire", "auriculaire"}


def script():
    """(doigts tendus, centre x, rayon) pour chaque image."""
    out = [None] * 35                                          # fond seul (apprentissage)
    out += [(DOIGTS5, 330, 45)] * 45                           # main ouverte immobile 1,5 s → pause
    out += [(set(), 330, 45 + 10 * k / 40) for k in range(40)]  # poing qui s'approche → volume
    out += [({"index", "majeur"}, 330, 45)] * 25                # main neutre
    out += [({"index", "majeur"}, 330 - 260 * k / 9, 45) for k in range(10)]   # balayage rapide
    out += [({"index", "majeur"}, 70, 45)] * 20
    return out


class Horloge:
    t = 1000.0

    def __call__(self):
        return self.t


class FausseCamera:
    def __init__(self, idx, horloge, shift=0):
        self.frames, self.i, self.h, self.shift = script(), 0, horloge, shift

    def set(self, *a):
        return True

    def isOpened(self):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        if self.shift == 0:
            self.i += 1
            self.h.t += 1 / 30
        else:
            self.i += 1
        img = BG.copy()
        if f is not None:
            tendus, x, r = f
            # caméra 2 : disparité d = f·B/Z = B·r/R (B = 10 cm, R = 4 cm) → la main approchée se décale plus
            dx = int(round(10.0 * r / 4.0)) if self.shift else 0
            m = main_synthetique(tendus, R=r, centre=(int(x) - dx, 250), rng=np.random.default_rng(0))
            img[m > 0] = SKIN
        return True, img

    def release(self):
        pass


def lancer(it, extra=(), touches=None, clics=None):
    h = Horloge()
    cams = []
    souris = {}
    touches = touches or {}

    def VideoCapture(idx):
        cams.append(FausseCamera(idx, h, shift=0 if not cams else 1))
        return cams[-1]

    n = {"k": 0}

    def waitKey(_):
        n["k"] += 1
        if clics and n["k"] == clics[0]:
            for x, y in clics[1]:
                souris["cb"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return touches.get(n["k"], -1)

    patches = {"VideoCapture": VideoCapture, "imshow": lambda *a: None, "waitKey": waitKey,
               "destroyWindow": lambda *a: None, "destroyAllWindows": lambda: None,
               "setMouseCallback": lambda w, cb: souris.__setitem__("cb", cb)}
    orig = {k: getattr(cv2, k) for k in patches}
    orig_time = app.time.monotonic
    sortie = []
    import builtins
    orig_print = builtins.print
    try:
        for k, v in patches.items():
            setattr(cv2, k, v)
        app.time.monotonic = h
        builtins.print = lambda *a, **kw: sortie.append(" ".join(str(x) for x in a))
        app.main(["--iteration", str(it), *extra])
    finally:
        for k, v in orig.items():
            setattr(cv2, k, v)
        app.time.monotonic = orig_time
        builtins.print = orig_print
    return sortie


def kinds(lignes):
    out = []
    for l in lignes:
        for k in ("play_pause", "volume", "seek_bloque", "seek"):
            if f"] {k}" in l:
                out.append(k)
                break
    return out


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        P = lambda n: os.path.join(d, n)                       # noqa: E731
        commun = ["--zone", P("zone.json"), "--calib", P("mono.npz"), "--stereo", P("stereo.npz"),
                  "--palm", P("palm.json"), "--log", P("events.csv")]
        res = {}
        for it in (1, 2, 3, 4, 5):
            res[it] = kinds(lancer(it, commun))
        # 6 : on clique une feuille qui couvre la main (à l'écran : HG, HD, BD, BG), image 20
        feuille = [(150, 60), (560, 60), (600, 470), (110, 470)]
        res[6] = kinds(lancer(6, commun, touches={19: ord("z")}, clics=(20, feuille)))
        zone_ok = os.path.exists(P("zone.json"))
        # 6 bis : zone loin de la main → aucune commande
        petite = [(10, 10), (60, 10), (60, 60), (10, 60)]
        res["6 hors zone"] = kinds(lancer(6, commun, touches={19: ord("z")}, clics=(20, petite)))
        os.remove(P("zone.json"))
        # 7 : touche c pendant le poing (image 110) → R étalonné et enregistré
        sortie7 = lancer(7, commun, touches={110: ord("c")})
        res[7] = kinds(sortie7)
        palm_ok = os.path.exists(P("palm.json"))
        # 8 : fausse paire stéréo (base 10 cm, cam2 décalée de la disparité)
        R, T = stereo_pair(10.0, 0.0)
        save_stereo(P("stereo.npz"), {"K1": camera_K(), "d1": np.zeros(5), "K2": camera_K(), "d2": np.zeros(5),
                                      "R": R, "T": T, "rms": 0.1, "size": (640, 480)})
        sortie8 = lancer(8, commun + ["--cam2", "1"])
        res[8] = kinds(sortie8)
        stereo_ok = any("Stéréo active" in l for l in sortie8)
        with open(P("events.csv"), encoding="utf-8") as f:
            rows = list(csv.reader(f))
        sessions = len(set(r[0] for r in rows[1:]))

    print("Commandes émises par main.py (caméra simulée) :")
    for k, v in res.items():
        print(f"  --iteration {k!s:<12}: {v}")
    print(f"Zone cliquée et enregistrée : {zone_ok};  R étalonné (touche c) : {palm_ok};  stéréo active : {stereo_ok}")
    print(f"Journal CSV : {len(rows) - 1} lignes, {sessions} sessions avec commandes (modes 5, 6, 7, 8)")
    assert res[1] == [] and res[2] == []
    assert res[3] == ["play_pause", "seek"], res[3]
    for it in (4, 5, 6, 7, 8):
        assert res[it][0] == "play_pause" and "volume" in res[it] and res[it][-1] == "seek", (it, res[it])
    assert res["6 hors zone"] == []
    assert zone_ok and palm_ok and stereo_ok and sessions == 4
    print("OK")
