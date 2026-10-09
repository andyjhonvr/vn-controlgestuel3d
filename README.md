python3 -m venv .venv
source .venv/bin/activate


pip install -r requirements.txt
python3 main.py                    # vidéo synthétique
python3 main.py --video film.mp4   # vraie vidéo (--cam 1 pour l'iPhone)

# Tests synthétiques
python3 tests/test_iter2_doigts.py
python3 tests/test_iter3_gestes.py
python3 tests/test_iter4_volume.py

Allow acces to camera