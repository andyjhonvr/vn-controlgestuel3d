"""Itération 5 : gestes robustes (anti-retour, réarmement, tolérance, journal CSV).
"""
import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from g3d.config import GestureConfig  # noqa: E402
from g3d.gestures import Event, EventLogger, GestureEngine  # noqa: E402
from test_iter3_gestes import W, trajectoire  # noqa: E402

ESSAIS = 100


def jouer(samples, robust):
    eng = GestureEngine(W, GestureConfig(), robust=robust)
    evs = [e for e in (eng.update(s) for s in samples) if e is not None]
    return evs, eng


def balayage_retour(rng):
    """Balayage droite (écran) en 0,3 s, arrêt 0,2 s, retour naturel en 0,35-0,8 s."""
    x0, y0 = 470 + rng.normal(0, 20), 250 + rng.normal(0, 20)
    x1 = x0 - 0.45 * W * rng.uniform(0.85, 1.15)
    segs = [(0.3, (x0, y0), (x0, y0), 2), (0.3, (x0, y0), (x1, y0), 2),
            (0.2, (x1, y0), (x1, y0), 2), (rng.uniform(0.35, 0.8), (x1, y0), (x0, y0), 2),
            (0.5, (x0, y0), (x0, y0), 2)]
    return trajectoire(segs, rng)


def balayage_puis_immobile(rng, poing=False):
    """Balayage main ouverte, puis main ouverte immobile 1,5 s (avec un poing avant si demandé)."""
    x0, y = 470 + rng.normal(0, 20), 250.0
    x1 = x0 - 0.45 * W
    segs = [(0.3, (x0, y), (x0, y), 5), (0.3, (x0, y), (x1, y), 5)]
    if poing:
        segs.append((0.4, (x1, y), (x1, y), 0))
    segs.append((1.5, (x1, y), (x1, y), 5))
    return trajectoire(segs, rng, bruit_px=2.0)


def sortie_puis_retour(rng):
    """Balayage droite, la main sort du champ 0,5 s, puis balayage gauche depuis l'arrivée."""
    x0, y = 470.0, 250.0
    x1 = x0 - 0.45 * W
    segs = [(0.3, (x0, y), (x0, y), 2), (0.3, (x0, y), (x1, y), 2), (0.5, (x1, y), (x1, y), -1),
            (0.3, (x1, y), (x1, y), 2), (0.3, (x1, y), (x0, y), 2), (0.3, (x0, y), (x0, y), 2)]
    return trajectoire(segs, rng)


def deux_balayages_droite(rng):
    """Balayage droite, retour LENT (1,5 s, pas un balayage), puis nouveau balayage droite."""
    x0, y = 470.0, 250.0
    x1 = x0 - 0.45 * W
    segs = [(0.3, (x0, y), (x0, y), 2), (0.3, (x0, y), (x1, y), 2), (1.5, (x1, y), (x0, y), 2),
            (0.3, (x0, y), (x0, y), 2), (0.3, (x0, y), (x1, y), 2), (0.3, (x1, y), (x1, y), 2)]
    return trajectoire(segs, rng)


def retour_apres_delai(rng):
    """Balayage droite, main tenue 2,7 s à l'arrivée, puis balayage gauche voulu."""
    x0, y = 470.0, 250.0
    x1 = x0 - 0.45 * W
    segs = [(0.3, (x0, y), (x0, y), 2), (0.3, (x0, y), (x1, y), 2), (2.7, (x1, y), (x1, y), 2),
            (0.3, (x1, y), (x0, y), 2), (0.3, (x0, y), (x0, y), 2)]
    return trajectoire(segs, rng)


def pause_scintillante(rng):
    """Main ouverte immobile 1,5 s; deux images isolées sans main (masque qui clignote)."""
    p = (320, 250)
    s = trajectoire([(1.5, p, p, 5)], rng, bruit_px=2.0)
    for i in (10, 20):
        s[i] = None
    return s


def pause_coupee(rng):
    """Main ouverte 0,5 s, main absente 3 images, main ouverte 0,5 s : la main est sortie."""
    p = (320, 250)
    return trajectoire([(0.5, p, p, 5), (0.1, p, p, -1), (0.5, p, p, 5)], rng, bruit_px=2.0)


def kinds(evs):
    return [(e.kind, e.value) for e in evs]


SCENARIOS = [
    ("balayage + retour naturel → 1 seul +10 s", balayage_retour,
     lambda e: kinds(e) == [("seek", 10.0)]),
    ("balayage main ouverte puis immobile → aucune pause", balayage_puis_immobile,
     lambda e: kinds(e) == [("seek", 10.0)]),
    ("balayage, poing, main ouverte immobile → +10 s puis pause",
     lambda r: balayage_puis_immobile(r, poing=True),
     lambda e: kinds(e) == [("seek", 10.0), ("play_pause", None)]),
    ("balayage, sortie du champ, balayage inverse → +10 s, -10 s", sortie_puis_retour,
     lambda e: kinds(e) == [("seek", 10.0), ("seek", -10.0)]),
    ("2 balayages droite (retour lent) → +10 s, +10 s", deux_balayages_droite,
     lambda e: kinds(e) == [("seek", 10.0), ("seek", 10.0)]),
    ("balayage inverse après 2,7 s → +10 s, -10 s", retour_apres_delai,
     lambda e: kinds(e) == [("seek", 10.0), ("seek", -10.0)]),
    ("main ouverte avec 2 images sans main → 1 pause", pause_scintillante,
     lambda e: kinds(e) == [("play_pause", None)]),
    ("main absente 3 images de suite → rien (sortie)", pause_coupee, lambda e: e == []),
]


def comparer():
    print(f"Gestes robustes ({ESSAIS} essais par scénario) :      avant (it. 3-4)   après (it. 5)")
    res = {}
    for nom, fab, ok in SCENARIOS:
        out = []
        for robust in (False, True):
            rng = np.random.default_rng(2026)
            n, faux = 0, 0
            for _ in range(ESSAIS):
                evs, _ = jouer(fab(rng), robust)
                n += ok(evs)
                faux += max(0, len(evs) - 1) if nom.startswith("balayage + retour") else 0
            out.append((n, faux))
        res[nom] = out
        extra = ""
        if nom.startswith("balayage + retour"):
            extra = f"   (faux retours : {out[0][1]} → {out[1][1]})"
        print(f"  {nom:<58} {out[0][0]:3d}/{ESSAIS}  →  {out[1][0]:3d}/{ESSAIS}{extra}")
    return res


def test_journal():
    """Les balayages bloqués sont mémorisés et le CSV contient la raison chiffrée."""
    rng = np.random.default_rng(1)
    eng = GestureEngine(W, GestureConfig())
    evs = []
    for s in balayage_retour(rng):
        e = eng.update(s)
        if e is not None:
            evs.append(e)
    bloques = list(eng.suppressed)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "results", "events_log.csv")
        log = EventLogger(path)
        for e in evs + bloques:
            log.log(e)
        log.log(Event("volume", 1.0, 62.0, "poing, test"))
        log.close()
        EventLogger(path).close()                        # 2e session : pas de 2e en-tête
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
    print(f"Journal CSV : {len(rows) - 1} lignes, en-tête {rows[0]}")
    for r in rows[1:]:
        print(f"    {r[2]:<12} {r[3]:>5}  {r[4][:90]}")
    assert rows[0] == EventLogger.HEADER and len(rows) == 1 + len(evs) + len(bloques) + 1
    assert any(r[2] == "seek_bloque" and "anti-retour" in r[4] for r in rows[1:])
    return True


if __name__ == "__main__":
    res = comparer()
    test_journal()
    apres = [v[1][0] for v in res.values()]
    assert min(apres) >= 98, "taux de réussite insuffisant"
    print("OK")
