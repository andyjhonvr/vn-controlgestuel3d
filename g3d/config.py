"""Paramètres du système. Tous les seuils sont regroupés ici pour être
faciles à justifier dans le rapport et à ajuster lors des essais."""
from dataclasses import dataclass, field


@dataclass
class SegmentationConfig:
    # Plages de chrominance de la peau dans YCrCb (Y n'est pas seuillé : robustesse à l'éclairage)
    cr_min: int = 133
    cr_max: int = 173
    cb_min: int = 77
    cb_max: int = 127
    # Soustraction d'arrière-plan par moyenne glissante
    bg_alpha: float = 0.02              # taux d'oubli du fond (0 = figé, 1 = remplacé à chaque image)
    bg_warmup_frames: int = 30          # ~1 s d'apprentissage du fond au démarrage et après 'b'
    bg_diff_thresh: int = 25            # écart minimal en niveaux de gris pour être « avant-plan »
    use_background: bool = True
    # Morphologie et filtrage du contour
    morph_kernel: int = 5               # taille du noyau elliptique (px)
    min_hand_area_ratio: float = 0.01   # aire minimale de la main / aire de l'image
    # Choix de la main parmi les taches de peau : on préfère celle qui touche le bord
    # gauche, droit ou bas de l'image (le bras y entre); le visage, lui, n'y touche pas.
    track_max_jump_ratio: float = 0.25  # saut max. du centre de paume entre deux images / largeur
    # Centre de paume quand le bras entre dans l'image : on parcourt la silhouette depuis
    # le bout de la main vers le bras; la paume est le premier élargissement suivi d'un
    # rétrécissement (le poignet) d'au moins wrist_dip_ratio.
    wrist_dip_ratio: float = 0.85       # le poignet est plus étroit que 85 % de la paume
    palm_min_ratio: float = 0.5         # la paume fait au moins 50 % du plus grand disque inscrit
    profile_bin_px: int = 4             # pas du profil de largeur (px)


@dataclass
class Config:
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)