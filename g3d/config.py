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
    # La silhouette DOIT toucher le bord gauche, droit ou bas via l'avant-bras.
    # Une tache flottante (ex. visage en mouvement) n'est jamais une main candidate.
    track_max_jump_ratio: float = 0.25  # distance max. ancien centre / nouveau contour, rapportée à la largeur
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
class GestureConfig:
    mirror: bool = True                   # sens des balayages tel que vu à l'écran (miroir)
    cooldown_s: float = 1.0               # délai après un balayage ou une pause
    # Balayage : |Δx| ≥ 25 % de la largeur en ≤ 0,5 s, avec |Δy| ≤ 0,5 |Δx|
    swipe_min_dx_ratio: float = 0.25
    swipe_max_duration_s: float = 0.5
    swipe_max_dy_ratio: float = 0.5
    seek_step_s: float = 10.0
    # Main ouverte immobile : ≥ 4 doigts, déplacement ≤ 15 px pendant 0,8 s
    open_min_fingers: int = 4
    still_max_px: float = 15.0
    still_duration_s: float = 0.8
    # Réarmement de la pause : après une pause, la main doit se fermer ou bouger de plus
    # de 45 px; après un balayage (itération 5), seulement se fermer (poing).
    pause_rearm_px: float = 45.0
    # Itération 5 : anti-retour. Pendant 2,5 s après un balayage, le sens opposé est
    # bloqué s'il part du côté où le balayage précédent s'est terminé (par rapport à son
    # point de départ). Sortir de la zone (ou perdre la main) réarme les deux sens.
    antireturn_s: float = 2.5
    # Itération 5 : jusqu'à 2 images consécutives sans main sont ignorées (masque qui
    # clignote); au-delà, la main est considérée sortie → reset().
    missing_tolerance_frames: int = 2
    # Volume : poing = embrayage; volume = V₀ + gain × (Z₀ − Z), moins la zone morte
    volume_gain_pct_per_cm: float = 2.0
    volume_median_n: int = 9              # médiane glissante sur Z
    volume_dead_zone_cm: float = 1.0      # on retranche 1 cm de |Z₀ − Z| (continu)
    volume_hysteresis_pct: float = 2.0    # on n'applique un changement que s'il est ≥ 2 %
    fist_tolerance_frames: int = 5        # images « pas poing » tolérées avant de lâcher
    volume_initial_pct: float = 50.0


@dataclass
class DepthConfig:
    # Profondeur monoculaire par le modèle sténopé : Z = f · R_réel / r_image.
    # f provisoire (webcam 640 px, champ horizontal ≈ 65°) : remplacée à l'itération 7.
    focal_px: float = 500.0
    palm_radius_cm: float = 4.0           # rayon du disque inscrit dans la paume / le poing
    # Itération 7 : étalonnage en un point du rayon réel R (touche c, poing à 50 cm)
    palm_calib_distance_cm: float = 50.0
    palm_calib_frames: int = 15           # médiane du rayon sur 15 images


@dataclass
class ZoneConfig:
    # Itération 6 : feuille lettre (8,5 × 11 po)
    sheet_long_cm: float = 27.94
    sheet_short_cm: float = 21.59
    # "paysage" : le grand côté est horizontal à l'écran; "portrait"; ou "auto" (déduit des
    # longueurs cliquées, peu fiable si la feuille est vue très en biais)
    orientation: str = "paysage"
    margin_cm: float = 0.0                # tolérance autour de la feuille
    # Itération 8 (stéréo) : la zone devient 3D, Z doit être dans [z_min, z_max]
    z_min_cm: float = 25.0
    z_max_cm: float = 90.0


@dataclass
class CalibrationConfig:
    # Itération 7 : damier imprimé, nombre de coins INTÉRIEURS (colonnes, lignes)
    pattern: tuple = (9, 6)
    square_cm: float = 2.5                # côté d'une case, mesuré à la règle
    min_views: int = 10                   # images minimales pour étalonner
    max_rms_px: float = 0.5               # critère de réussite


@dataclass
class PlayerConfig:
    synthetic_duration_s: float = 180.0   # durée de la vidéo synthétique (sans --video)
    width: int = 640
    height: int = 360
    message_s: float = 2.0                # durée d'affichage du dernier événement


@dataclass
class Config:
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)
    fingers: FingerConfig = field(default_factory=FingerConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    calib: CalibrationConfig = field(default_factory=CalibrationConfig)
