# CLAUDE.md — Contrôle gestuel avec commande de profondeur 3D (GIF-7001)

> Mémoire du projet pour Claude Code. Ce fichier résume tout le travail fait jusqu'ici, les décisions prises et la suite. **Le code actuel est dans ce dépôt : lis-le avant de proposer des changements.**

---

## 1. Qui nous sommes et ce que nous attendons de toi

Nous sommes une équipe de 5 étudiants du cours **GIF-7001 Vision numérique (Université Laval, automne 2026)**. Le responsable technique est Andy Vasquez Ramos. Nous construisons un système qui pilote un lecteur vidéo par gestes de la main, avec une **profondeur réelle de la main mesurée par triangulation stéréoscopique**. Agis comme ingénieur principal en vision numérique et comme soutien technique de l'équipe.

**Langue :** tout est en **français** : réponses, code, docstrings, commentaires, messages de commit, rapport et vidéo.

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
└── docs/  (etalonnage.md, guion_video.md, rapport, backlog, plan des itérations)
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

**Documents déjà produits (à ranger dans `docs/`) :**
- Rapport préliminaire (Word, 10 pages, 8 figures). Il manque le lien de la vidéo.
- Backlog de 30 histoires utilisateur (Word).
- Prompt maître du projet (Markdown pour Notion).
- Plan des itérations (Word).
- Diagrammes d'activité (démarrage, boucle par image, règles du moteur).

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
| 3 | Je contrôle une vidéo (MVP) | Balayage = ±10 s, main ouverte immobile = pause | ⏭️ **Suivante** |
| 4 | Volume par la profondeur (mono) | Approcher le poing fait monter le volume | À faire |
| 5 | Gestes robustes | Anti-retour, réarmement, tolérance, CSV des raisons | À faire |
| 6 | Zone de contrôle | Feuille lettre cliquée; rien ne se déclenche hors de la zone | À faire |
| 7 | Étalonnage | Damier, K, RMS < 0,5 px; Z mono avec la vraie focale | À faire |
| 8 | Stéréo 3D | Z réelle avec deux caméras, zone 3D | À faire (après les sem. 11–13) |
| 9 | Expériences et figures | `run_all.py` régénère les 7 figures + `summary.json` | En parallèle |

**Itération 2 (implémentée) :**
- `FingerConfig` : `max_defect_angle_deg=90`, `min_defect_depth_ratio=0.15`, `protrusion_ring=1.6`, `protrusion_max_chord=0.9`, `protrusion_min_reach=2.0`.
- `count_fingers` : un creux est un défaut de convexité de profondeur ≥ 0,15 r et d'angle ≤ 90°, **dont les deux bords sont à ≥ 1,7 r du centre de paume** (`tip_min_ratio`); doigts = bouts distincts (fusion à 0,5 r, `tip_merge_ratio`). Sans creux, une protubérance étroite compte pour 1 doigt; sinon, c'est un poing (0).
- `HandObservation` contient `fingers`, `finger_valleys`, `hull` et `solidity`. `main.py` affiche `doigts : n (POING/OUVERTE/autre)` et la solidité.
- Tests synthétiques (180 mains avec avant-bras, rotations, main gauche et droite) : 100 % exact en conditions nominales; 83 % exact et 100 % sur les classes utiles en conditions difficiles.
- **Écarté :** un filtre qui ignorait les creux « du côté du bras » dégradait le comptage quand le pouce pointe vers le poignet (5 → 4).
- **Corrigé :** `defects.reshape(-1, 4)`, à cause de la différence de forme entre versions d'OpenCV.
- **Corrigé (webcam : 4 → 5, 3 → 4) :** filtre des bouts de doigts. `tests/test_iter2_doigts.py` (160 mains avec doigts et pouce repliés) : 80,0 % → 99,4 % exact; conditions difficiles (bords bruités, doigts serrés) : 78,9 % → 91,4 % exact, classes utiles 86,7 % → 99,2 %.
- **Reste à faire :** confirmer avec la webcam et noter la solidité observée avec le poing et avec la main ouverte.

**Itération 3 (suivante) :**
- `g3d/gestures.py` : `Sample`, `Event` et `GestureEngine`, avec balayage, main ouverte immobile (sans réarmement pour l'instant) et délai de 1 s. `mirror=True`.
- `g3d/player.py` : `VideoPlayer`, avec vidéo réelle ou synthétique, `handle(event)` et surimpression.
- `main.py` : intègre le moteur et le lecteur, et affiche chaque événement avec sa raison.
- Anti-retour, réarmement, volume et CSV arrivent aux itérations 4 et 5.

---

## 6. Problèmes déjà résolus (à ne pas répéter)

| Symptôme | Cause | Solution |
|---|---|---|
| Poing lu comme 1 doigt | Le poignet allonge la silhouette | Un doigt = protubérance **étroite** (corde < 0,9 r) |
| Le point rouge part vers l'avant-bras | L'avant-bras contient un disque aussi grand que la paume | Profil de demi-largeur depuis le bout de la main; paume = premier sommet avant le poignet (baisse ≥ 15 %). Testé : 24/24 contre 14/24 |
| Le point part sur le visage et y reste | Fond appris sur la 1re image; le masque n'était jamais mis à jour; on prenait la plus grande tache | Apprentissage de 30 images; main = tache qui touche le bord gauche, droit ou bas (+ suivi); fond mis à jour sauf sur la main confirmée. Testé : main choisie 70/70 contre 0/70 |
| Le retour de la main déclenche un recul | Mouvement naturel après un balayage | Anti-retour de 2,5 s |
| Pause parasite après un balayage | Main ouverte immobile à la fin du balayage | Réarmement seulement quand la main se ferme |
| Volume qui tremble ou biaisé | Bruit sur Z et erreurs de comptage | Médiane de 9, hystérésis de 2 %, tolérance de 5 images |
| Un doigt de trop (4 → 5, 3 → 4, poing → 3) | Le creux entre un doigt tendu et une jointure ou le pouce replié comptait (« n creux → n + 1 ») | Les deux bords du creux doivent être à ≥ 1,7 r de la paume; on compte les bouts distincts. 80 % → 99,4 % |
| Plantage `cannot unpack non-iterable numpy.int32` | `convexityDefects` renvoie `(N,4)` selon la version d'OpenCV | `defects.reshape(-1, 4)` |

---

## 7. Au début de chaque session

1. Lis ce fichier et le code de `g3d/` et de `main.py`.
2. Confirme l'état en 3 ou 4 lignes : quelle itération est terminée, laquelle est en validation, laquelle vient ensuite.
3. **N'écris pas de code** tant qu'on ne te le demande pas (« donne-moi l'itération N » ou une correction précise).
4. À la fin de chaque itération, **mets à jour les sections 5 et 6 de ce fichier** (état et problèmes résolus) et propose un message de commit en français.
