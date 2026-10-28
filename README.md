python3 -m venv .venv
source .venv/bin/activate


pip install -r requirements.txt
python3 main.py                    # vidéo synthétique
python3 main.py --video film.mp4   # vraie vidéo (--cam 1 pour l'iPhone)

# Tests synthétiques (sans caméra)
python3 tests/test_iter1_visage.py
python3 tests/test_iter2_doigts.py
python3 tests/test_iter3_gestes.py
python3 tests/test_iter4_volume.py
python3 tests/test_iter5_robuste.py
python3 tests/test_iter6_zone.py
python3 tests/test_iter7_etalonnage.py
python3 tests/test_iter8_stereo.py
python3 tests/test_main_simule.py      # main.py, modes 1 à 8, caméra simulée

# Expériences : 7 figures + results/summary.json (~1 min)
python3 experiments/run_all.py

Allow acces to camera

## Eviter la selection du visage en mouvement

La selection exige maintenant une silhouette reliee au bord gauche, droit ou bas
par l'avant-bras, a chaque image. Si ce lien disparait, le programme indique
« aucune main » et suspend les gestes au lieu de se rabattre sur le visage.
Garder l'avant-bras denude et visible jusqu'au bord; une manche ou une coupure
dans le masque peut donc interrompre le suivi. Garder la main separee du visage
dans l'image. Un visage raccorde au bras dans le masque ou touchant lui-meme un
bord autorise reste un cas non resolu par cette regle geometrique.

Pour valider : relancer le programme, retirer la main et appuyer sur `b`, puis
bouger la tete sans main pendant 10 s (aucun point rouge ni commande attendus).
Introduire ensuite la main par le bas et bouger la tete : le point rouge doit
rester sur la paume. Retirer la main : retour a « aucune main ». Le visage peut
encore apparaitre en blanc dans le masque de diagnostic, qui montre les pixels
de peau en mouvement; cela ne signifie pas qu'il est selectionne pour les gestes.
Si le point rouge reste sur le visage, verifier que sa silhouette ne touche pas
un bord autorise et ne rejoint pas la main ou l'avant-bras.

Regression synthetique : `python -X utf8 tests/test_iter1_visage.py`.

## Filmer uniquement l'itération 1

Dans PyCharm, clic droit sur `tests/iteration1.py` puis **Run 'iteration1'**.
En terminal : `python tests/iteration1.py` (ou `python main.py --iteration 1`).
L'option `--cam 1` permet de sélectionner une autre caméra.

Ce mode affiche la caméra et le masque, le contour vert, le centre et le cercle
rouges de la paume, ainsi que l'entrée du bras en jaune. Il ne calcule pas les
doigts ni la profondeur, et ne lance ni les commandes gestuelles ni le lecteur.
Gardez la main hors du champ pendant l'apprentissage du fond, puis faites entrer
l'avant-bras par le bas ou un côté. Vérifiez le suivi et le retour à « aucune main »
après retrait. Touches : `b` fond, `d` masque, `m` diagnostic, `q` quitter.
Si le fond est mal appris, retirez la main et appuyez sur `b`.

`main.py` sans option lance toutes les fonctions (itérations 1 à 8; profondeur mono
si aucune 2e caméra n'est donnée).

## Filmer l'itération 2 sans les commandes des itérations 3 et 4

Dans PyCharm, clic droit sur `tests/iteration2.py`, puis
**Run 'iteration2'**. En terminal depuis la racine du projet : `python tests/iteration2.py`, ou
`python main.py --iteration 2`. L'option `--cam N` reste disponible.

Ce mode conserve le suivi, le comptage des doigts, les classes POING / OUVERTE /
autre, la solidité, l'enveloppe convexe et les creux magenta. Il n'instancie ni
lecteur ni moteur de gestes et ne calcule pas la profondeur. Seules Camera et
Masque s'ouvrent au démarrage; Diagnostic reste accessible avec `m`.

Pour tester, garder la main hors du champ pendant l'apprentissage, puis montrer
successivement 0 à 5 doigts à environ 50 cm. Ça fonctionne si le nombre et la
classe correspondent à la position, sans fenêtre Lecteur ni indication Z ou
VOLUME. Si le comptage est instable, vérifier le masque avec `m`, écarter les doigts
et garder la paume face à la caméra; si le fond est mal appris, retirer la main et
appuyer sur `b`. Les touches `d`, `m`, `b`, `q` restent disponibles.

## Filmer l'itération 3 sans la profondeur et le volume

Dans PyCharm, clic droit sur `tests/iteration3.py`, puis **Run 'iteration3'**.
En terminal depuis la racine du projet : `python tests/iteration3.py`, ou
`python main.py --iteration 3`. Les options `--cam N` et `--video fichier.mp4`
restent disponibles. Sans fichier vidéo, le lecteur synthétique s'ouvre.

Ce mode conserve le suivi, les doigts, les balayages de ±10 s et la commande
lecture/pause. Il ne calcule pas Z et n'appelle pas la commande de volume; la barre
de volume est masquée. Le poing reste neutre et réarme la pause.

Pour tester : retirer la main pendant l'apprentissage, puis présenter 4 ou 5 doigts
immobiles pendant environ 1 s pour la pause. Fermer le poing puis rouvrir la main
et l'immobiliser pour reprendre. Avec 2 doigts, balayer horizontalement au moins
un quart de l'image en moins de 0,5 s : droite à l'écran = +10 s, gauche = -10 s.
Attendre 2 s entre les essais. Ça fonctionne si le lecteur affiche PAUSE / LECTURE
ou ±10 s, sans Z ni volume. Vérifier l'horloge à distance des extrémités de la vidéo.
Si un geste échoue, vérifier d'abord le nombre de doigts, puis l'immobilité ou
l'amplitude et la vitesse du balayage. L'anti-retour reste prévu pour l'itération 5.

## Filmer l'itération 4 avec la profondeur et le volume

Dans PyCharm, clic droit sur `tests/iteration4.py`, puis **Run 'iteration4'**.
En terminal depuis la racine du projet : `python tests/iteration4.py`, ou
`python main.py --iteration 4`. Les options `--cam N` et `--video fichier.mp4`
restent disponibles. Ce lanceur active la version complète des itérations 1 à 4,
comme `main.py` sans option : doigts, lecture/pause, balayages, profondeur et volume.

Pour tester : laisser la main hors champ pendant l'apprentissage, puis fermer
le poing à environ 50 cm et attendre environ 1 s. Vérifier `0 (POING)` et la ligne
verte `VOLUME (poing)`. Approcher lentement le poing sans le tourner : Z doit
diminuer et le volume augmenter. En l'éloignant, Z augmente et le volume diminue.
L'avant-bras doit rester visible jusqu'au bord, sans rejoindre le visage dans l'image.

La barre verte du lecteur représente le volume; aucun son n'est joué. Z est une
estimation monoculaire avec une focale provisoire, pas une distance étalonnée.
Le poing doit rester fermé pour régler le volume; pendant sa prise, les balayages
et la pause sont bloqués. Main retirée, le volume est conservé. Rouvrir la main
relâche la prise après la tolérance de quelques images et permet les autres gestes.

Le volume démarre à 50 % à chaque lancement. Pour comparer plusieurs essais depuis
50 %, relancer le programme entre les essais; `b` réapprend seulement le fond.
Si le réglage échoue, vérifier d'abord `POING`, le masque (`m`) et la continuité du
bras. Si le volume bouge au repos, noter les variations de Z pour le protocole de
bruit avant de modifier les seuils. Test automatique :
`python -X utf8 tests/test_iter4_volume.py`.


## Filmer l'itération 5 : gestes robustes

`python tests/iteration5.py` (ou `python main.py --iteration 5`). Mêmes gestes qu'à
l'itération 4, avec : **anti-retour** (le retour naturel de la main après un balayage
est bloqué 2,5 s et affiché `seek_bloque` dans le terminal), **réarmement** (après un
balayage, la pause n'est possible qu'après avoir fermé le poing), **tolérance** de 2
images sans main, et **journal CSV** `results/events_log.csv` (une ligne par commande
ou blocage, avec la raison chiffrée).

Pour tester : balayer vers la droite puis ramener la main normalement → un seul +10 s.
Après un balayage, laisser la main ouverte immobile → pas de pause; fermer le poing puis
rouvrir → pause. Sortir la main du champ réarme les deux sens.

## Filmer l'itération 6 : zone de contrôle

`python tests/iteration6.py`. Appuyer sur `z` et cliquer les 4 coins de la feuille lettre
(haut-gauche, haut-droit, bas-droit, bas-gauche, tels que vus à l'écran). La feuille est
dessinée en vert (rouge si la main est hors zone) et la position de la main sur la feuille
est affichée en cm. Hors de la feuille : `HORS ZONE`, aucune commande. `x` efface la zone.

## Filmer l'itération 7 : étalonnage

1. `python scripts/capture_calibration.py --square 2.5` : 15-20 vues du damier (espace),
   puis `c` → `data/mono.npz`, RMS < 0,5 px.
2. `python tests/iteration7.py` : la ligne bleue affiche `f=… px (etalonnee)`. Poing à 50 cm,
   touche `c` → R étalonné (`data/palm.json`). Z est alors en vrais centimètres.

## Filmer l'itération 8 : stéréo

1. `python scripts/capture_calibration.py --cam 0 --cam2 1 --square 2.5` → `data/stereo.npz`.
2. `python tests/iteration8.py` (2e caméra = 1 par défaut, sinon `--cam2 N`).
   Z triangulé (`Z st`), fenêtre `Camera 2`, zone 3D (feuille + Z entre 25 et 90 cm).

## Itération 9 : expériences et figures

`python tests/iteration9.py` (ou `python experiments/run_all.py`) régénère
`results/fig1` à `fig7` et `results/summary.json`. Avec des vraies mains enregistrées par
`python scripts/record_dataset.py` (touches 0-5), une 8e figure est ajoutée.
