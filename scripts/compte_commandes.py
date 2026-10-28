"""Compte les commandes de la dernière session de results/events_log.csv.

Lancer juste après avoir quitté main.py (ou un lanceur tests/iterationN.py) avec q :
    python scripts/compte_commandes.py
Chaque lancement du programme est une « session » : on ne compte que la dernière.
Option : --toutes pour afficher le total de chaque session.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "results" / "events_log.csv"))
    ap.add_argument("--toutes", action="store_true", help="résumé de toutes les sessions")
    args = ap.parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"Aucun journal ({path}) : lancer d'abord l'itération 5 ou plus.")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Journal vide : aucune commande enregistrée.")

    if args.toutes:
        for s, n in Counter(r["session"] for r in rows).items():
            print(f"{s}  {n} ligne(s)")
        return
    s = rows[-1]["session"]
    c = Counter(f"{r['type']} {r['valeur']}".strip() for r in rows if r["session"] == s)
    print("Session", s)
    for k, v in sorted(c.items()):
        print(f"  {k:<16} {v}")
    print(f"  {'total':<16} {sum(c.values())}")


if __name__ == "__main__":
    main()
