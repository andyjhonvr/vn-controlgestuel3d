"""Lancer ce fichier dans PyCharm pour tester uniquement l'itération 1."""
import sys

from main import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--iteration", "1"])
