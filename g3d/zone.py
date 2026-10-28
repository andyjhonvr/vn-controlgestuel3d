"""Zone de contrôle (itération 6) : une feuille lettre cliquée dans l'image.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ZoneConfig
from .geometry import apply_homography, dlt_homography


class ControlZone:
    def __init__(self, corners_px: Sequence[Sequence[float]], size_cm: Tuple[float, float],
                 cfg: ZoneConfig | None = None):
        self.cfg = cfg or ZoneConfig()
        self.corners = np.asarray(corners_px, np.float64).reshape(4, 2)   # image brute
        self.size_cm = (float(size_cm[0]), float(size_cm[1]))
        w, h = self.size_cm
        self.sheet = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        self.H = dlt_homography(self.corners, self.sheet)                 # image → feuille

    # ------------------------------------------------------------ création
    @classmethod
    def from_clicks(cls, clicks_screen, frame_width: int, cfg: ZoneConfig | None = None,
                    mirror: bool = True) -> "ControlZone":
        """Clics faits sur l'affichage miroir (HG, HD, BD, BG à l'écran) → zone."""
        cfg = cfg or ZoneConfig()
        pts = np.asarray(clicks_screen, np.float64).reshape(4, 2).copy()
        if mirror:
            pts[:, 0] = frame_width - 1 - pts[:, 0]
        # Orientation de la feuille (config) ou, en mode "auto", côté le plus long dans l'image
        landscape = cfg.orientation == "paysage"
        if cfg.orientation == "auto":
            top = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
            side = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
            landscape = top >= side
        size = ((cfg.sheet_long_cm, cfg.sheet_short_cm) if landscape
                else (cfg.sheet_short_cm, cfg.sheet_long_cm))
        return cls(pts, size, cfg)

    # ------------------------------------------------------------ requêtes
    def to_sheet(self, x: float, y: float) -> Tuple[float, float]:
        u, v = apply_homography(self.H, (x, y))
        return float(u), float(v)

    def contains(self, x: float, y: float, z_cm: Optional[float] = None, use_z: bool = False) -> bool:
        m = self.cfg.margin_cm
        u, v = self.to_sheet(x, y)
        w, h = self.size_cm
        inside = -m <= u <= w + m and -m <= v <= h + m
        # l'homographie d'un point très loin peut changer de signe : on vérifie aussi le polygone
        inside = inside and cv2.pointPolygonTest(self.corners.astype(np.float32), (float(x), float(y)),
                                                 True) >= -self._margin_px()
        if inside and use_z:
            inside = z_cm is not None and self.cfg.z_min_cm <= z_cm <= self.cfg.z_max_cm
        return bool(inside)

    def _margin_px(self) -> float:
        # marge en px ≈ marge en cm × (périmètre px / périmètre cm)
        per_px = sum(np.linalg.norm(self.corners[i] - self.corners[i - 1]) for i in range(4))
        return self.cfg.margin_cm * per_px / (2 * sum(self.size_cm))

    # ---------------------------------------------------------- persistance
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"corners_px": self.corners.tolist(), "size_cm": list(self.size_cm),
                       "H_image_vers_feuille": self.H.tolist(),
                       "z_min_cm": self.cfg.z_min_cm, "z_max_cm": self.cfg.z_max_cm}, f, indent=2)

    @classmethod
    def load(cls, path: str, cfg: ZoneConfig | None = None) -> Optional["ControlZone"]:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cfg = cfg or ZoneConfig()
        cfg.z_min_cm = float(d.get("z_min_cm", cfg.z_min_cm))
        cfg.z_max_cm = float(d.get("z_max_cm", cfg.z_max_cm))
        return cls(d["corners_px"], d["size_cm"], cfg)

    # ------------------------------------------------------------- dessin
    def draw(self, img: np.ndarray, inside: Optional[bool] = None) -> None:
        """Dessine la feuille sur l'image BRUTE (avant le miroir d'affichage)."""
        color = (0, 200, 0) if inside in (None, True) else (0, 0, 255)
        cv2.polylines(img, [self.corners.astype(np.int32)], True, color, 2)
