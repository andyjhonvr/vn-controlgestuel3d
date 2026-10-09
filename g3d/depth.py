"""Profondeur de la main.

Itération 4 : estimation monoculaire. Dans le modèle sténopé, un objet de taille réelle
R à la distance Z apparaît avec une taille r = f · R / Z (en pixels). On prend comme
objet le disque inscrit dans la paume (ou le poing), dont le rayon r est déjà mesuré
par la transformée de distance : Z = f · R / r.

L'erreur relative sur Z est celle sur r (dZ/Z = −dr/r) : à 60 cm, r ≈ 33 px, et une
erreur de 1 px donne déjà ≈ 1,8 cm. D'où la médiane et la zone morte du moteur de gestes.
La stéréo (itération 8) remplacera cette estimation.
"""
from __future__ import annotations

from typing import Optional

from .config import DepthConfig


def mono_depth_cm(palm_radius_px: float, cfg: DepthConfig | None = None) -> Optional[float]:
    """Distance caméra–main (cm) à partir du rayon de paume en pixels."""
    cfg = cfg or DepthConfig()
    if palm_radius_px is None or palm_radius_px < 1.0:
        return None
    return cfg.focal_px * cfg.palm_radius_cm / palm_radius_px


def radius_px_at(z_cm: float, cfg: DepthConfig | None = None) -> float:
    """Inverse de mono_depth_cm : rayon attendu (px) à la distance z_cm (sert aux tests)."""
    cfg = cfg or DepthConfig()
    return cfg.focal_px * cfg.palm_radius_cm / z_cm
