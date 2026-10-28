"""Expérience 5 : gestes (100 essais par scénario) et volume stéréo / mono.
"""
import numpy as np

from common import BLEU, GRIS, ORANGE, plt, save
from g3d.config import GestureConfig
from g3d.gestures import GestureEngine, Sample
import test_iter3_gestes as t3
import test_iter5_robuste as t5

ESSAIS = 100


def _taux(fab, ok, robust):
    rng = np.random.default_rng(2026)
    n, lat, faux = 0, [], 0
    for _ in range(ESSAIS):
        out = fab(rng)
        samples, t0 = out if isinstance(out, tuple) else (out, None)
        eng = GestureEngine(t3.W, GestureConfig(), robust=robust)
        evs = [e for e in (eng.update(s) for s in samples) if e is not None]
        if ok(evs):
            n += 1
            if evs and t0 is not None:
                lat.append(evs[0].t - t0)
        faux += max(0, len(evs) - 1)
    return n, (1000 * float(np.mean(lat)) if lat else None), faux


def scenarios():
    return [
        ("balayage droite 0,3 s", t3.balayage(+1), t3.un_seul("seek", +10)),
        ("balayage gauche 0,3 s", t3.balayage(-1), t3.un_seul("seek", -10)),
        ("main ouverte immobile", t3.immobile(5), t3.un_seul("play_pause")),
        ("pause, poing, pause", t3.pause_poing_pause, lambda e: [x.kind for x in e] == ["play_pause"] * 2),
        ("déplacement lent (2 s) → rien", t3.balayage(+1, 2.0), t3.aucun),
        ("diagonale → rien", t3.balayage(+1, 0.3, dy=1.0), t3.aucun),
        ("poing immobile → rien", t3.immobile(0), t3.aucun),
        ("3 doigts immobiles → rien", t3.immobile(3), t3.aucun),
    ] + [(nom.split(" → ")[0], fab, ok) for nom, fab, ok in t5.SCENARIOS]


def volume_essais(sigma, n=100, seed=5):
    rng = np.random.default_rng(seed)
    fins, courbe = [], None
    zs = np.concatenate([np.full(18, 50.0), 50 - 10 * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 30))),
                         np.full(30, 40.0)])
    for k in range(n):
        eng = GestureEngine(640, GestureConfig())
        vols, zb = [], []
        for i, z in enumerate(zs):
            zm = z + rng.normal(0, sigma)
            eng.update(Sample(i / 30, 320, 250, 0, zm))
            vols.append(eng.volume)
            zb.append(zm)
        fins.append(eng.volume)
        if k == 0:
            courbe = (np.arange(len(zs)) / 30, zs, np.array(zb), np.array(vols))
    return float(np.mean(fins)), float(np.std(fins)), courbe


def run(sigma_z_50=None):
    sigma_z_50 = sigma_z_50 or {"stereo": 0.56, "mono": 1.03}
    rows = []
    for nom, fab, ok in scenarios():
        av = _taux(fab, ok, robust=False)
        ap = _taux(fab, ok, robust=True)
        rows.append((nom, av, ap))
    fig, ax = plt.subplots(figsize=(8, 5.6))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[1][0] for r in rows], 0.4, color=GRIS, label="avant (itérations 3-4)")
    ax.barh(y - 0.2, [r[2][0] for r in rows], 0.4, color=BLEU, label="après (itération 5)")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"essais réussis sur {ESSAIS}")
    ax.set_xlim(0, 105)
    ax.set_title("Gestes sur trajectoires synthétiques (30 im/s, bruit 2-3 px)")
    ax.legend(loc="lower right", fontsize=8)
    f6 = save(fig, "fig6_gestes.png")

    vol = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for nom, col in (("stereo", BLEU), ("mono", ORANGE)):
        m, s, (t, zt, zm, v) = volume_essais(sigma_z_50[nom])
        vol[nom] = {"sigma_z_cm": round(sigma_z_50[nom], 3), "volume_final_pct": round(m, 2),
                    "ecart_type_pct": round(s, 2)}
        axs[0].plot(t, zm, color=col, alpha=0.7, lw=1, label=f"Z mesuré {nom} (σ = {sigma_z_50[nom]:.2f} cm)")
        axs[1].plot(t, v, color=col, lw=2, label=f"{nom} : {m:.1f} ± {s:.1f} % (100 essais)")
    axs[0].plot(t, zt, "k--", lw=1, label="Z vrai")
    axs[0].set_xlabel("temps (s)")
    axs[0].set_ylabel("Z (cm)")
    axs[0].set_title("Poing approché de 50 à 40 cm")
    axs[0].legend(fontsize=8)
    axs[1].axhline(68, color="k", ls="--", lw=1, label="attendu 68 %")
    axs[1].set_xlabel("temps (s)")
    axs[1].set_ylabel("volume (%)")
    axs[1].set_title("Volume commandé")
    axs[1].legend(fontsize=8)
    f7 = save(fig, "fig7_volume.png")

    retour = next(r for r in rows if r[0].startswith("balayage + retour"))
    return {"scenarios": {r[0]: {"avant": r[1][0], "apres": r[2][0],
                                 "delai_ms": None if r[2][1] is None else round(r[2][1])} for r in rows},
            "balayage_retour_faux_avant": retour[1][2], "balayage_retour_faux_apres": retour[2][2],
            "volume": vol, "figures": [f6, f7]}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False))
