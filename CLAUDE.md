# CLAUDE.md — Contrôle gestuel avec commande de profondeur 3D (GIF-7001)

> Mémoire du projet pour Claude Code. Ce fichier résume tout le travail fait jusqu'ici, les décisions prises et la suite. **Le code actuel est dans ce dépôt : lis-le avant de proposer des changements.**

**Façon de travailler (obligatoire) :**
1. On travaille **par itérations**. Tu n'implémentes l'itération N que lorsqu'on te dit « donne-moi l'itération N ».
2. **Tu modifies les fichiers directement dans le dépôt** et tu montres les diffs pour qu'on les accepte. Avant de modifier, tu indiques en 2 ou 3 lignes quels fichiers tu vas toucher et pourquoi.
3. **Tu testes avant de dire que c'est terminé** : tu écris et exécutes des tests sur des images ou des trajectoires synthétiques dans `tests/` (scripts simples lancés avec `python3`, pytest non obligatoire), et tu donnes l'avant / après en chiffres.
4. **Tu ne vois pas la webcam.** Tu peux lancer `main.py`, mais la validation visuelle est faite par un membre de l'équipe; il te décrit ce qu'il voit et tu corriges.
5. Face à un problème : d'abord la **cause profonde** (en bref), puis la correction, puis « ce que tu dois voir ».
6. Chaque livraison se termine par **comment tester**, **ça fonctionne si…** et **si ça échoue…** (symptôme → paramètre à ajuster).
7. N'ajoute jamais de dépendance autre qu'opencv-python, numpy et matplotlib.

**Environnement :** macOS, zsh, environnement virtuel `.venv` (exécution avec `python3`). L'avertissement macOS `AVCaptureDeviceTypeExternal is deprecated` est sans conséquence; si l'image vient de l'iPhone (Continuity Camera), lancer `python3 main.py --cam 1`. Les versions d'OpenCV diffèrent sur la forme de certains résultats (par exemple `convexityDefects` renvoie `(N,1,4)` ou `(N,4)`) : toujours utiliser des `reshape` défensifs.

---

## 2. Contexte du projet

| Élément | Détail |
|---|---|
| Équipe | Agbogan Kokou Joel (537 419 445) · Boucher Dominic (536 888 890) · Daigle Benoit (537 002 267) · Roman Vincent Théo (537 470 712) · Vasquez Ramos Andy Jhonell (537 398 376) |
| Rapport préliminaire | Remise le **20 oct. 2026**. Grille : page titre 1, contexte et solution 2, justification 1, résultats 1, étapes 2, vidéo 1, langue 2 |
| Rapport final | Semaine 15 (vers le 8 déc. 2026, à confirmer) |
| Matière utile | Étalonnage et homographie (sem. 3–5), traitement des images (6–7, 10–11), stéréoscopie (11–13) |

**Gestes :**

| Geste | Règle | Commande |
|---|---|---|
| Main ouverte immobile | 4 doigts ou plus, déplacement ≤ 15 px pendant 0,8 s | Lecture / pause |
| Balayage droite / gauche | \|Δx\| ≥ 25 % de la largeur en ≤ 0,5 s, \|Δy\| ≤ 0,5 \|Δx\| | ±10 s |
| Poing + rapprocher / éloigner | 0 doigt; volume = V₀ + 2 %/cm × (Z₀ − Z) | Volume |

**Contraintes :**
- **Aucun apprentissage automatique** (ni MediaPipe, ni réseau de neurones, ni cascade de Haar).
- Seulement `opencv-python`, `numpy` et `matplotlib`, avec Python 3.10 ou plus récent.
- Tous les seuils sont dans `g3d/config.py`.
- Les calculs se font sur l'**image brute**; le miroir ne sert qu'à l'affichage et au sens des balayages (`mirror=True`).
- Graines aléatoires fixes dans les expériences.

---

## 3. Architecture convenue

```
projet/
├── requirements.txt  .gitignore  README.md  main.py  CLAUDE.md
├── g3d/  __init__.py  config.py  hand.py  geometry.py  calibration.py
│         zone.py  depth.py  gestures.py  player.py
├── scripts/  capture_calibration.py  record_dataset.py
├── experiments/  exp_hand.py  exp_calibration.py  exp_homography.py
│                 exp_triangulation.py  exp_gestures.py  exp_real_hands.py  run_all.py
├── tests/  (tests synthétiques de chaque itération)
├── data/  (mono.npz, stereo.npz, zone.json, calib_images/, real_hands/)
├── results/  (figures, summary.json, events_log.csv)
```

**Flux à chaque image :** `HandSegmenter.segment` → `extract_hand` → `ControlZone.contains` → Z (`StereoRig.triangulate` ou `mono_depth_cm`) → `Sample` → `GestureEngine.update` → `Event` → `VideoPlayer.handle`.

**Ordre fixe des règles dans `GestureEngine.update` :**
1. Aucune main ou main hors zone → `reset()`.
2. **Volume** (commande continue, sans délai) : poing; tolérance de 5 images mal comptées; médiane de 9 valeurs de Z; zone morte de 1 cm; hystérésis de 2 %.
3. **Délai** (cooldown) de 1 s.
4. **Balayage** avec **règle anti-retour** : le sens opposé est bloqué pendant 2,5 s s'il part du côté où le balayage précédent s'est terminé; sortir de la zone réarme les deux sens.
5. **Main ouverte immobile** avec **réarmement** : après une pause, la règle se réarme si la main se ferme ou se déplace de plus de 45 px; après un balayage, seulement si la main se ferme.

Chaque `Event` porte sa raison chiffrée.

---

## 4. Ce qui existe déjà

Une **version de référence complète** a été développée dans une conversation précédente sur claude.ai : les 9 modules, `main.py`, le script d'étalonnage et les 5 expériences. Ses résultats sur données synthétiques sont les **objectifs** à reproduire :

| Expérience | Résultat |
|---|---|
| Doigts, conditions nominales (240 images) | 100 %; centre de paume à 1,5 px près |
| Doigts, conditions difficiles | 76 % exact; 95 % sur les classes fonctionnelles (poing / ouverte / autre) |
| Étalonnage synthétique | RMS 0,075 px; fx +0,008 % |
| Homographie de la zone | σ = 1 px → 0,13 cm; σ = 3 px → 0,41 cm; DLT = OpenCV |
| Triangulation (30 → 80 cm) | Z stéréo 0,34 → 2,41 cm; mono 2,65 → 7,43 cm |
| Gestes (100 essais) | Balayages 100 % (200 ms); pause 100 % (800 ms); balayage + retour 99 % (22 fausses détections sans anti-retour); volume stéréo 67,8 ± 1,0 %, mono 68,7 ± 3,9 % |

**À propos du comptage des doigts :** il sert à lire l'*intention* de la main, et non sa position.
- Le **poing** est « l'embrayage » du volume : le volume ne bouge que tant que le poing est fermé.
- La **main ouverte** rend la pause possible.
- Les autres formes sont neutres.
- Seules ces trois classes comptent. La **solidité** (aire de la main / aire de son enveloppe convexe) reste une solution de secours si le comptage est instable.

---

## 5. Plan des itérations et état

| # | Itération | Ce qu'on doit voir | État |
|---|---|---|---|
| 1 | Je vois ma main | Contour, centre de paume et entrée du bras qui suivent la main; le visage disparaît | ✅ Terminée |
| 2 | Je compte les doigts | 0 à 5 doigts à l'écran, creux marqués, poing = 0 | 🔎 Implémentée; **validation avec la webcam en cours** |
| 3 | Je contrôle une vidéo (MVP) | Balayage = ±10 s, main ouverte immobile = pause | 🔎 Implémentée; **validation avec la webcam à faire** |
| 4 | Volume par la profondeur (mono) | Approcher le poing fait monter le volume | 🔎 Implémentée; **validation avec la webcam à faire** |
| 5 | Gestes robustes | Anti-retour, réarmement, tolérance, CSV des raisons | 🔎 Implémentée; **validation avec la webcam à faire** |
| 6 | Zone de contrôle | Feuille lettre cliquée; rien ne se déclenche hors de la zone | 🔎 Implémentée; **validation avec la webcam à faire** |
| 7 | Étalonnage | Damier, K, RMS < 0,5 px; Z mono avec la vraie focale | 🔎 Implémentée; **étalonnage réel à faire** (damier imprimé) |
| 8 | Stéréo 3D | Z réelle avec deux caméras, zone 3D | 🔎 Implémentée; **validation avec deux caméras à faire** |
| 9 | Expériences et figures | `run_all.py` régénère les 7 figures + `summary.json` | ✅ Terminée (synthétique); mains réelles à enregistrer |

**Lanceurs de démonstration :** `tests/iteration1.py` à `tests/iteration9.py` (un par
itération, chacun ajoute la fonction nouvelle aux précédentes). `main.py` sans option =
itération 8 (mono si pas de `--cam2` ni de `data/stereo.npz`). Les modes `--iteration 1`
à `4` gardent exactement leur comportement (moteur `robust=False`, focale provisoire, pas
de zone).
`tests/test_main_simule.py` lance `main.py` en modes 1 à 8 avec une caméra simulée
(touches, clics de zone, touche `c`, fausse paire stéréo) : OK.

**Mode de démonstration de l'itération 1 :**
Correction de la sélection du visage en mouvement : `_select_contour` exige
désormais une liaison au bord gauche/droit/bas à chaque image; suppression du
repli sur la plus grande tache et du suivi des taches flottantes. La distance
maximale de suivi s'applique aussi aux candidats reliés au bord. Si aucun candidat
valide n'existe, `None` réinitialise les gestes via la boucle existante.
`tests/test_iter1_visage.py` : sur 120 scènes, visage seul et visage proche de
l'ancien point sélectionnés 120/120 avant → 0/120 après; main + visage correctement
suivis 120/120 avant et après; retrait de main correctement reconnu 0/120 → 120/120.
Chaîne couleur/fond/gestes : visage mobile visible dans le masque sur 90 images,
mais 0 main sélectionnée et 0 commande après correction. Comptage : 159/160,
classes utiles 100 %; tests de volume : OK. Validation webcam restant à faire.
Limite : avant-bras continu jusqu'au bord nécessaire; la règle ne sépare pas une
silhouette fusionnée main/visage et n'identifie pas un visage touchant un bord admis.

Avant les validations suivantes, on peut maintenant filmer l'itération 1 seule :
clic droit sur `iteration1.py` dans PyCharm → Run, ou `python main.py --iteration 1`.
Ce mode conserve la segmentation, la paume et l'entrée du bras, sans comptage des
doigts, profondeur, moteur de gestes ni lecteur. `main.py` sans option reste complet.
Validation du mode isolé : masque synthétique, géométrie identique au mode complet,
absence d'appels aux fonctions désactivées, touches et fermeture testées avec caméra
simulée. Tests existants 2, 3 et 4 : OK sous Python 3.10 avec `-X utf8` sous Windows.
La validation visuelle du nouveau mode à la webcam reste à faire par l'utilisateur.

**Itération 2 (implémentée) :**
- Mode isolé : lancer `tests/iteration2.py` dans PyCharm ou `python main.py --iteration 2`.
  Affiche le comptage, les classes, la solidité et les creux, sans lecteur, gestes ni Z.
  Modes 1, 2 et mode complet vérifiés avec caméra simulée (touches et fermeture incluses).
  Test synthétique des doigts : 159/160 exact, classes utiles 100 %; validation webcam à faire.
- `FingerConfig` : `max_defect_angle_deg=90`, `min_defect_depth_ratio=0.15`, `protrusion_ring=1.6`, `protrusion_max_chord=0.9`, `protrusion_min_reach=2.0`.
- `count_fingers` : un creux est un défaut de convexité de profondeur ≥ 0,15 r et d'angle ≤ 90°, **dont les deux bords sont à ≥ 1,7 r du centre de paume** (`tip_min_ratio`); doigts = bouts distincts (fusion à 0,5 r, `tip_merge_ratio`). Sans creux, une protubérance étroite compte pour 1 doigt; sinon, c'est un poing (0).
- `HandObservation` contient `fingers`, `finger_valleys`, `hull` et `solidity`. `main.py` affiche `doigts : n (POING/OUVERTE/autre)` et la solidité.
- Tests synthétiques (180 mains avec avant-bras, rotations, main gauche et droite) : 100 % exact en conditions nominales; 83 % exact et 100 % sur les classes utiles en conditions difficiles.
- **Écarté :** un filtre qui ignorait les creux « du côté du bras » dégradait le comptage quand le pouce pointe vers le poignet (5 → 4).
- **Corrigé :** `defects.reshape(-1, 4)`, à cause de la différence de forme entre versions d'OpenCV.
- **Corrigé (webcam : 4 → 5, 3 → 4) :** filtre des bouts de doigts. `tests/test_iter2_doigts.py` (160 mains avec doigts et pouce repliés) : 80,0 % → 99,4 % exact; conditions difficiles (bords bruités, doigts serrés) : 78,9 % → 91,4 % exact, classes utiles 86,7 % → 99,2 %.
- **Reste à faire :** confirmer avec la webcam et noter la solidité observée avec le poing et avec la main ouverte.

**Itération 3 (implémentée) :**
- Mode isolé : `tests/iteration3.py` dans PyCharm, ou `python main.py --iteration 3`.
  Suivi, doigts, lecture/pause et balayages actifs; calcul de Z, commande de volume
  et barre de volume désactivés. Le poing reste neutre et réarme la pause.
  `GestureEngine(..., volume_enabled=False)` saute entièrement `_volume` et vide
  la trajectoire lorsque le poing est fermé. `VideoPlayer(..., show_volume=False)`
  masque la barre. Les valeurs par défaut préservent le mode complet.
  Validation sans webcam : 12 scénarios × 100 essais passent avec le volume désactivé;
  balayage 0,3 s = 140 ms et pause = 833 ms, comme avant. Poing mobile sans commande,
  affichage et touches des modes 1 à 4, options du lanceur vérifiés. Tests de volume
  de l'itération 4 : OK. Validation visuelle du mode 3 à faire par l'utilisateur.
- `g3d/gestures.py` : `Sample(t, x, y, fingers, z_cm, in_zone)`, `Event(kind, t, value, reason)` et `GestureEngine(frame_width, cfg)`. Ordre : reset → volume → délai de 1 s → balayage → main ouverte immobile. `mirror=True` : une main qui va vers la gauche de l'image brute est un balayage **droite** à l'écran (+10 s).
- Main ouverte immobile : ≥ 4 doigts sur toute la fenêtre de 0,8 s, et écart max. à la position médiane ≤ 15 px. **Verrou simple** (le réarmement complet viendra à l'itération 5) : après une pause, il faut un poing ou un déplacement de plus de 45 px.
- `g3d/player.py` : `VideoPlayer(path)` lit une vraie vidéo (`--video`) ou une vidéo synthétique de 3 min (horloge et disque qui fait un tour toutes les 10 s). Le volume est une barre, car OpenCV ne joue pas le son. Touche espace = lecture / pause manuelle.
- `main.py` : fenêtres « Camera » et « Lecteur »; chaque événement est affiché avec sa raison et imprimé dans le terminal.
- `tests/test_iter3_gestes.py` (12 scénarios × 100 essais, 30 im/s, bruit de 2 à 3 px) : 100 % partout. Délais : balayage de 0,3 s → 140 ms, pause → 833 ms. Aucune fausse détection pour un mouvement lent (2 s), une diagonale, une dérive de 60 px/s, un poing ou 3 doigts.
- Limite connue de la règle : une dérive lente (moins de ≈ 35 px/s) compte comme immobile, et un mouvement ample de 1,5 s peut atteindre 25 % de la largeur en 0,5 s.

**Itération 4 (implémentée) :**
- Lanceur de démonstration : `tests/iteration4.py`, avec le même principe que les
  lanceurs précédents. Force `--iteration 4` et transmet `--cam` / `--video`.
  Active les fonctions cumulées 1 à 4; aucune modification des algorithmes.
  Affiche Z, la prise du poing et la barre de volume (sans son). Le volume initial
  est de 50 % à chaque lancement; `b` ne le réinitialise pas.
  Vérification : lancement depuis `tests` et transmission des options OK;
  `test_iter4_volume.py` OK (68,2 ± 1,9 % à sigma Z = 1,5 cm, chaîne complète
  65,2 % comme précédemment). La validation webcam reste à faire.
- `g3d/depth.py` : `mono_depth_cm(r_px) = f · R / r`, avec `DepthConfig` : `focal_px=500` (provisoire jusqu'à l'itération 7) et `palm_radius_cm=4.0`.
- Volume dans `GestureEngine` : le poing (0 doigt) sert d'embrayage; tolérance de 5 images; Z₀ = première médiane de 9 valeurs; volume = V₀ + 2 %/cm × (ΔZ moins 1 cm de zone morte **continue**); hystérésis de 2 %; bornes 0–100 %. Pendant la prise du poing, aucun balayage ni aucune pause; le poing réarme la pause.
- `tests/test_iter4_volume.py` :
  - Z mono sur des poings synthétiques (30 → 80 cm) : biais brut de −6 à −7 % (f et R sont provisoires); après étalonnage en un point, erreur ≤ 0,44 cm.
  - Volume 50 → 40 cm (attendu 68 %) : 67,6 ± 0,8 % (σ_Z = 0,5 cm), 68,2 ± 1,9 % (1,5 cm), 68,4 ± 3,9 % (3 cm). Sans filtres : 62 événements au lieu de 7.
  - Poing immobile pendant 3 s : volume modifié dans 0/100 essais (σ = 0,5 cm), 1/100 (1 cm), 28/100 (1,5 cm).
  - Chaîne complète image → volume : 65,2 % (attendu 68 %), à cause du biais de f.
- **Reste à faire (webcam) :** noter le bruit de Z avec le poing immobile; si le volume tremble, augmenter `volume_dead_zone_cm` (1,5 cm → 9/100 à σ = 1,5 cm).

**Itération 5 (implémentée) :**
- `GestureEngine(..., robust=True)` (défaut; `robust=False` = itérations 3-4 à l'identique).
- Anti-retour (`antireturn_s=2,5`) : un balayage de sens opposé est bloqué s'il commence
  du côté de l'arrivée par rapport au **point de départ** du balayage précédent. Le blocage
  vide la trajectoire et est mémorisé dans `engine.suppressed` (type `seek_bloque`, avec raison).
  `reset()` (main sortie ou hors zone) réarme les deux sens.
- Réarmement : après un balayage, verrou de type « balayage » levé seulement par un poing;
  après une pause, poing ou déplacement > 45 px (inchangé).
- Tolérance : `missing_tolerance_frames=2` images sans main ignorées; à la 3e, `reset()`
  (le test « main perdue 3 images → rien » de l'itération 3 reste vrai).
- `EventLogger` : `results/events_log.csv` (`session, t_s, type, valeur, raison`), en ajout.
- `tests/test_iter5_robuste.py` (8 scénarios × 100, avant → après) : balayage + retour naturel
  23 → 100 (77 faux retours → 0); balayage puis main ouverte immobile sans pause 0 → 100;
  main ouverte avec 2 images manquantes → pause 0 → 100; les 5 autres scénarios 100 → 100.
  Tests 1 à 4 : chiffres identiques.

**Itération 6 (implémentée) :**
- `g3d/geometry.py` : DLT normalisée (Hartley) et triangulation DLT, écrites à la main.
- `g3d/zone.py` : `ControlZone.from_clicks` (clics sur l'affichage miroir, ordre HG, HD, BD, BG
  à l'écran), `contains(x, y, z, use_z)`, `to_sheet` (cm), `data/zone.json`.
  `ZoneConfig.orientation="paysage"` (le mode `auto` se trompe 43/200 fois en forte perspective).
- `main.py` : touche `z` (cliquer), `x` (effacer), `Échap`; feuille verte / rouge; `Sample.in_zone`.
- `tests/test_iter6_zone.py` : DLT = OpenCV (3·10⁻⁵ mm sans bruit, 0,35 mm à σ = 1 px car OpenCV
  raffine par LM); position sur la feuille σ = 1 px → 0,10 cm, 3 px → 0,29 cm; contains
  4000/4000; gestes hors zone 0 commande 100/100; sortie de zone réarme l'anti-retour 100/100.
- **Corrigé :** le 4e clic arrivant pendant `waitKey` faisait afficher un 5e coin
  (`IndexError`) → la zone est finalisée au début de l'image suivante.

**Itération 7 (implémentée) :**
- `g3d/calibration.py` (coins sous-pixel, `calibrate_mono`, `calibrate_stereo`, npz),
  `scripts/capture_calibration.py` (espace = garder, `c` = étalonner, `--from-images`).
- `main.py --iteration 7` : f = fx de `data/mono.npz` (mise à l'échelle de la largeur);
  touche `c` = R étalonné avec le poing à 50 cm (médiane de 15 rayons) → `data/palm.json`.
- `tests/test_iter7_etalonnage.py` (damiers rendus avec K, distorsion connus, f = 600) :
  RMS 0,021 px, fx +0,006 % (difficile : 0,092 px, −0,08 %); Z mono 30 → 80 cm :
  erreur moyenne 12,4 cm (f = 500, R = 4) → 0,32 cm; chaîne volume 61,7 % → 67,2 % (attendu 68).

**Itération 8 (implémentée) :**
- `StereoRig` (`g3d/depth.py`) : points non distordus → coordonnées normalisées → DLT.
  `main.py --cam2 N` : 2e segmentation, Z triangulé, fenêtre « Camera 2 », zone 3D
  (`z_min_cm=25`, `z_max_cm=90`). Sans `--cam2` ou `stereo.npz` : retour au mono.
- `tests/test_iter8_stereo.py` : étalonnage stéréo RMS 0,023 px, base 9,998 cm (10), rotation
  0,04°; DLT = cv2.triangulatePoints (6·10⁻¹³ cm); bruit 1 px, Z 30 → 80 cm : stéréo
  0,17 → 1,16 cm, mono 0,29 → 2,06 cm; chaîne image → Z stéréo 0,32 cm; zone 3D 6/6;
  volume 50 → 40 cm : stéréo 67,5 ± 0,9 %, mono 67,9 ± 1,5 %.

**Itération 9 (terminée en synthétique) :** `experiments/run_all.py` (≈ 35 s) → `results/fig1`
à `fig7` + `summary.json`; `exp_real_hands.py` ajoute `fig8` si `data/real_hands/` contient
des masques (`scripts/record_dataset.py`, touches 0-5). Résultats : doigts nominal 98,3 %,
difficile 87,5 % (classes 95,8 %); étalonnage RMS 0,021 px; zone 0,11 / 0,34 cm (σ = 1 / 3 px);
Z stéréo 0,17 → 1,16 cm, mono 0,29 → 2,06 cm; faux retours 77 → 0; volume stéréo
67,9 ± 0,8 %, mono 68,0 ± 1,2 %. Données synthétiques partagées : `experiments/synth.py`.

---

## 6. Problèmes déjà résolus (à ne pas répéter)

- Retour naturel après un balayage compté comme balayage inverse (77/100 essais) :
  anti-retour de 2,5 s relatif au point de départ du balayage précédent → 0/100.
- Pause parasite après un balayage main ouverte (100/100) : verrou « balayage » levé
  seulement par le poing → 0/100.
- Une image sans main (masque qui clignote) annulait une pause en cours : tolérance de
  2 images → 100/100 pauses.
- Orientation automatique de la feuille fausse en forte perspective (43/200) :
  orientation fixée dans `ZoneConfig` (paysage par défaut).
- `IndexError` à la fin des 4 clics de zone (clic reçu pendant `waitKey`) : zone
  finalisée au début de l'image suivante.
- `tests/iteration1.py` ne trouvait pas `main` hors PyCharm : ajout du `sys.path` comme
  dans les autres lanceurs.

- Lancement séparé de l'itération 4 dans PyCharm : ajout de `tests/iteration4.py`,
  utilisant le mode complet existant, sans copie des calculs ou des réglages.

- Visage mobile repris comme main après perte du bras : le repli sur la plus
  grande tache et le suivi sans liaison au bord étaient encore actifs. Tous les
  candidats doivent désormais toucher un bord d'entrée de bras; sinon aucune
  observation n'est envoyée. Le masque brut de peau/mouvement peut encore montrer
  le visage sans qu'il soit sélectionné. Régression : `tests/test_iter1_visage.py`.

- Le volume et la profondeur gênaient la démonstration de l'itération 3 : lanceur
  `tests/iteration3.py` et option `--iteration 3`, avec désactivation du calcul de
  profondeur et du moteur de volume, et masquage de la barre du lecteur.

- Les commandes de lecture et de volume gênaient les essais de comptage : mode
  `--iteration 2` et lanceur `tests/iteration2.py`, sans instanciation du lecteur
  ou du moteur de gestes et sans calcul de profondeur. Le lanceur retrouve la
  racine du projet même lorsqu'il est exécuté depuis le dossier `tests`.

- Démonstration de l'itération 1 encombrée par les autres fonctions : ajout du
  lanceur `iteration1.py` et de l'option `--iteration 1`; aucune commande du lecteur
  n'est exécutée dans ce mode. Si le suivi échoue, réapprendre le fond avec `b`,
  main hors champ, et faire entrer l'avant-bras par le bas ou un côté.

| Symptôme | Cause | Solution |
|---|---|---|
| Poing lu comme 1 doigt | Le poignet allonge la silhouette | Un doigt = protubérance **étroite** (corde < 0,9 r) |
| Le point rouge part vers l'avant-bras | L'avant-bras contient un disque aussi grand que la paume | Profil de demi-largeur depuis le bout de la main; paume = premier sommet avant le poignet (baisse ≥ 15 %). Testé : 24/24 contre 14/24 |
| Le point part sur le visage et y reste | Fond appris sur la 1re image; le masque n'était jamais mis à jour; on prenait la plus grande tache | Apprentissage de 30 images; main = tache qui touche le bord gauche, droit ou bas (+ suivi); fond mis à jour sauf sur la main confirmée. Testé : main choisie 70/70 contre 0/70 |
| Le retour de la main déclenche un recul | Mouvement naturel après un balayage | Anti-retour de 2,5 s |
| Pause parasite après un balayage | Main ouverte immobile à la fin du balayage | Réarmement seulement quand la main se ferme |
| Volume qui tremble ou biaisé | Bruit sur Z et erreurs de comptage | Médiane de 9, hystérésis de 2 %, tolérance de 5 images |
| Un doigt de trop (4 → 5, 3 → 4, poing → 3) | Le creux entre un doigt tendu et une jointure ou le pouce replié comptait (« n creux → n + 1 ») | Les deux bords du creux doivent être à ≥ 1,7 r de la paume; on compte les bouts distincts. 80 % → 99,4 % |
| La pause ne se réarmait jamais après un poing | Pendant la prise du poing, le moteur sortait avant de déverrouiller la pause | Déverrouillage avant la règle du volume. 0/100 → 100/100 |
| Volume qui bouge avec un poing immobile (98/100 essais à σ_Z = 1,5 cm) | Zone morte « sèche » : au premier dépassement de 1 cm, la cible sautait de ±2 % (= l'hystérésis) | Zone morte continue (on retranche 1 cm de ΔZ) : 98/100 → 28/100 à σ = 1,5 cm, 1/100 à σ = 1 cm |
| Plantage `cannot unpack non-iterable numpy.int32` | `convexityDefects` renvoie `(N,4)` selon la version d'OpenCV | `defects.reshape(-1, 4)` |

---

## 7. Au début de chaque session

1. Lis ce fichier et le code de `g3d/` et de `main.py`.
2. Confirme l'état en 3 ou 4 lignes : quelle itération est terminée, laquelle est en validation, laquelle vient ensuite.
3. **N'écris pas de code** tant qu'on ne te le demande pas (« donne-moi l'itération N » ou une correction précise).
4. À la fin de chaque itération, **mets à jour les sections 5 et 6 de ce fichier** (état et problèmes résolus) et propose un message de commit en français.
