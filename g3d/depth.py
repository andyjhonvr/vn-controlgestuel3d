"""Profondeur de la main.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import DepthConfig
from .geometry import triangulate_dlt


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


def calibrate_palm_radius(palm_radius_px: float, z_cm: float, focal_px: float) -> float:
    """Rayon réel R (cm) du disque de paume, mesuré avec le poing à une distance connue."""
    return z_cm * palm_radius_px / focal_px


def save_palm(path: str, palm_radius_cm: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"palm_radius_cm": palm_radius_cm}, f, indent=2)


def load_palm(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return float(json.load(f)["palm_radius_cm"])


class StereoRig:
    """Deux caméras étalonnées : x2 = R · x1 + T (T en cm). Repère = caméra 1."""

    def __init__(self, K1, d1, K2, d2, R, T, size: Tuple[int, int] = (640, 480)):
        self.K1, self.d1 = np.asarray(K1, np.float64), np.asarray(d1, np.float64)
        self.K2, self.d2 = np.asarray(K2, np.float64), np.asarray(d2, np.float64)
        self.R, self.T = np.asarray(R, np.float64), np.asarray(T, np.float64).reshape(3)
        self.size = size
        # Matrices de projection en coordonnées NORMALISÉES (distorsion déjà retirée)
        self.P1n = np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2n = np.hstack([self.R, self.T.reshape(3, 1)])

    @classmethod
    def from_calib(cls, c: Dict) -> "StereoRig":
        return cls(c["K1"], c["d1"], c["K2"], c["d2"], c["R"], c["T"], c.get("size", (640, 480)))

    @property
    def baseline_cm(self) -> float:
        return float(np.linalg.norm(self.T))

    def _normalized(self, pt, K, d) -> np.ndarray:
        p = np.asarray(pt, np.float64).reshape(1, 1, 2)
        return cv2.undistortPoints(p, K, d).reshape(2)

    def triangulate(self, pt1, pt2) -> np.ndarray:
        """Point 3D (cm, repère caméra 1) à partir des pixels dans les deux images."""
        x1 = self._normalized(pt1, self.K1, self.d1)
        x2 = self._normalized(pt2, self.K2, self.d2)
        return triangulate_dlt(self.P1n, self.P2n, x1, x2)

    def depth_cm(self, pt1, pt2) -> Optional[float]:
        if pt1 is None or pt2 is None:
            return None
        z = float(self.triangulate(pt1, pt2)[2])
        return z if z > 0 else None
