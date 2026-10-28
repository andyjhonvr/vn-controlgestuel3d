"""Régénère toutes les figures (results/fig1 … fig7) et results/summary.json.
"""
import json
import time

from common import RESULTS, ROOT
import exp_calibration
import exp_gestures
import exp_hand
import exp_homography
import exp_real_hands
import exp_triangulation


def main():
    t0 = time.time()
    summary = {}
    etapes = [("doigts_et_paume", exp_hand.run), ("etalonnage", exp_calibration.run),
              ("homographie_zone", exp_homography.run), ("triangulation", exp_triangulation.run)]
    for nom, fn in etapes:
        t = time.time()
        summary[nom] = fn()
        print(f"  {nom:<18} OK ({time.time() - t:4.1f} s)")
    t = time.time()
    summary["gestes_et_volume"] = exp_gestures.run(summary["triangulation"]["sigma_z_50cm"])
    print(f"  {'gestes_et_volume':<18} OK ({time.time() - t:4.1f} s)")
    summary["mains_reelles"] = exp_real_hands.run()
    print(f"  {'mains_reelles':<18} {'OK' if summary['mains_reelles']['disponible'] else 'sauté (aucune donnée)'}")
    summary["genere_le"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    figs = [p for v in summary.values() if isinstance(v, dict) for p in v.get("figures", [])]
    print(f"{len(figs)} figures et {(RESULTS / 'summary.json').relative_to(ROOT)} générés en "
          f"{time.time() - t0:.0f} s :")
    for p in figs:
        print("   ", p)

    # Résumé lisible
    s = summary
    d = s["doigts_et_paume"]
    print("\nRésumé :")
    print(f"  Doigts : nominal {d['nominal']['exact_pct']:.1f} % exact; difficile {d['difficile']['exact_pct']:.1f} % "
          f"exact, {d['difficile']['classes_utiles_pct']:.1f} % classes utiles")
    e = s["etalonnage"]["15_vues_nominal"]
    print(f"  Étalonnage (15 vues) : RMS {e['rms_px']:.3f} px, |fx| {e['fx_erreur_pct']:.3f} %")
    h = s["homographie_zone"]["erreur_cm"]
    print(f"  Zone : σ = 1 px → {h['sigma_1px']:.2f} cm; σ = 3 px → {h['sigma_3px']:.2f} cm")
    tr = s["triangulation"]
    print(f"  Z (30 → 80 cm) : stéréo {tr['stereo_erreur_moy_cm'][0]:.2f} → {tr['stereo_erreur_moy_cm'][-1]:.2f} cm; "
          f"mono {tr['mono_erreur_moy_cm'][0]:.2f} → {tr['mono_erreur_moy_cm'][-1]:.2f} cm")
    g = s["gestes_et_volume"]
    print(f"  Balayage + retour : faux retours {g['balayage_retour_faux_avant']} → {g['balayage_retour_faux_apres']}")
    v = g["volume"]
    print(f"  Volume (attendu 68 %) : stéréo {v['stereo']['volume_final_pct']:.1f} ± {v['stereo']['ecart_type_pct']:.1f} %, "
          f"mono {v['mono']['volume_final_pct']:.1f} ± {v['mono']['ecart_type_pct']:.1f} %")
    return summary


if __name__ == "__main__":
    main()
