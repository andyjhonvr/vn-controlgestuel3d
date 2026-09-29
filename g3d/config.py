"""Paramètres du système. Tous les seuils sont regroupés ici pour ajuster lors des essais."""
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
class FingerConfig:
    # Creux entre deux doigts = défaut de convexité profond et aigu
    max_defect_angle_deg: float = 90.0    # angle max. au fond du creux (loi des cosinus)
    min_defect_depth_ratio: float = 0.15  # profondeur min. du creux / rayon de paume
    # Un creux ne compte que si ses DEUX bords sont des bouts de doigts tendus, loin de la
    # paume; une jointure ou un pouce replié reste près de la paume (≈ 1,3 r).
    tip_min_ratio: float = 1.7            # distance min. bout de doigt – centre de paume / r
    tip_merge_ratio: float = 0.5          # deux bouts plus proches que 0,5 r = le même doigt
    # Sans aucun creux : un doigt isolé est une protubérance ÉTROITE qui sort d'un
    # cercle de protrusion_ring × r; le poignet, large, n'est pas compté.
    protrusion_ring: float = 1.6
    protrusion_max_chord: float = 0.9     # largeur max. d'un doigt au passage du cercle / r
    protrusion_min_reach: float = 2.0     # le doigt doit atteindre au moins 2 r du centre


@dataclass
class Config:
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)
    fingers: FingerConfig = field(default_factory=FingerConfig)