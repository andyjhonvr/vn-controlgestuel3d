"""Lancer ce fichier dans PyCharm pour tester la détection et le comptage des doigts."""
import sys
from pathlib import Path

# Permet aussi le lancement direct depuis le dossier tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--iteration", "2"])
