"""Itération 3 : balayages et main ouverte immobile sur des trajectoires synthétiques.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g3d.config import GestureConfig  # noqa: E402
from g3d.gestures import GestureEngine, Sample  # noqa: E402

W, FPS, ESSAIS = 640, 30.0, 100


def lisse(u):
    """Profil de vitesse en cloche (0 → 1), comme un geste humain."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


def trajectoire(segments, rng, bruit_px=3.0, doigts=2):
    """segments : liste de (durée_s, (x0, y0), (x1, y1), doigts ou None).
    Renvoie une liste de Sample ou None (main absente si doigts == -1)."""
    out, t = [], 0.0
    for dur, p0, p1, d in segments:
        n = max(1, int(round(dur * FPS)))
        for k in range(n):
            u = lisse((k + 1) / n)
            x = p0[0] + (p1[0] - p0[0]) * u + rng.normal(0, bruit_px)
            y = p0[1] + (p1[1] - p0[1]) * u + rng.normal(0, bruit_px)
            nd = doigts if d is None else d
            out.append(None if nd == -1 else Sample(t, x, y, nd))
            t += 1 / FPS
    return out


def jouer(samples, cfg=None):
    eng = GestureEngine(W, cfg or GestureConfig())
    evs = []
    for s in samples:
        e = eng.update(s)
        if e is not None:
            evs.append(e)
    return evs


def scenario(nom, fabrique, verifie):
    rng = np.random.default_rng(2026)
    ok, lat = 0, []
    for _ in range(ESSAIS):
        samples, t_debut = fabrique(rng)
        evs = jouer(samples)
        if verifie(evs):
            ok += 1
            if evs and t_debut is not None:
                lat.append(evs[0].t - t_debut)
    txt = f"  {nom:<52} {ok:3d}/{ESSAIS}"
    if lat:
        txt += f"   délai moyen {1000 * np.mean(lat):.0f} ms"
    print(txt)
    return ok / ESSAIS


def un_seul(kind, value=None):
    return lambda evs: len(evs) == 1 and evs[0].kind == kind and (value is None or evs[0].value == value)


def aucun(evs):
    return len(evs) == 0


def balayage(sens_ecran, duree=0.3, dy=0.0):
    def f(rng):
        x0 = 470 + rng.normal(0, 20)
        dx = -(0.45 * W) * sens_ecran * rng.uniform(0.85, 1.15)     # miroir : écran droite = brut gauche
        if sens_ecran < 0:
            x0 = 170 + rng.normal(0, 20)
        y0 = 250 + rng.normal(0, 20)
        segs = [(0.3, (x0, y0), (x0, y0), None),
                (duree, (x0, y0), (x0 + dx, y0 + dy * abs(dx)), None),
                (0.4, (x0 + dx, y0 + dy * abs(dx)), (x0 + dx, y0 + dy * abs(dx)), None)]
        return trajectoire(segs, rng), 0.3
    return f


def immobile(doigts, duree=2.5, derive_px_s=0.0, bruit=2.0):
    def f(rng):
        p0 = (320 + rng.normal(0, 30), 250 + rng.normal(0, 30))
        p1 = (p0[0] + derive_px_s * duree, p0[1])
        return trajectoire([(duree, p0, p1, doigts)], rng, bruit_px=bruit), 0.0
    return f


def pause_poing_pause(rng):
    p = (320, 250)
    segs = [(1.5, p, p, 5), (0.4, p, p, 0), (1.5, p, p, 5)]
    return trajectoire(segs, rng, bruit_px=2.0), 0.0


def pause_coupee(rng):
    """Main ouverte 0,5 s, main absente 3 images, main ouverte 0,5 s : pas de pause."""
    p = (320, 250)
    segs = [(0.5, p, p, 5), (0.1, p, p, -1), (0.5, p, p, 5)]
    return trajectoire(segs, rng, bruit_px=2.0), 0.0


if __name__ == "__main__":
    print("Gestes de l'itération 3 (trajectoires synthétiques, 30 im/s, bruit 2-3 px) :")
    r = []
    r.append(scenario("balayage droite (écran) en 0,3 s → +10 s", balayage(+1), un_seul("seek", +10)))
    r.append(scenario("balayage gauche (écran) en 0,3 s → -10 s", balayage(-1), un_seul("seek", -10)))
    r.append(scenario("balayage rapide en 0,15 s → +10 s", balayage(+1, 0.15), un_seul("seek", +10)))
    r.append(scenario("main ouverte immobile 2,5 s → 1 seule pause", immobile(5), un_seul("play_pause")))
    r.append(scenario("4 doigts immobiles → 1 seule pause", immobile(4), un_seul("play_pause")))
    r.append(scenario("pause, poing, pause → 2 pauses", pause_poing_pause,
                      lambda e: [x.kind for x in e] == ["play_pause", "play_pause"]))
    neg = []
    neg.append(scenario("déplacement lent (2 s) → rien", balayage(+1, 2.0), aucun))
    neg.append(scenario("déplacement en diagonale (|dy| = |dx|) → rien", balayage(+1, 0.3, dy=1.0), aucun))
    neg.append(scenario("main ouverte qui dérive (60 px/s) → rien", immobile(5, derive_px_s=60), aucun))
    neg.append(scenario("poing immobile → rien", immobile(0), aucun))
    neg.append(scenario("3 doigts immobiles → rien", immobile(3), aucun))
    neg.append(scenario("main ouverte interrompue (main perdue) → rien", pause_coupee, aucun))
    assert min(r + neg) >= 0.98, "taux de réussite insuffisant"
    print("OK")
