"""Données synthétiques à vérité terrain connue (étalonnage, zone, stéréo).
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

W, H = 640, 480


def camera_K(f: float = 600.0, cx: float = 322.0, cy: float = 238.0) -> np.ndarray:
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)


DIST = np.array([-0.12, 0.05, 0.0, 0.0, 0.0])          # distorsion radiale modérée (webcam)


def _texture(pattern, square_cm, ppc):
    cols, rows = pattern
    nx, ny = cols + 1, rows + 1                           # cases
    margin = 1.0                                          # marge blanche d'une case
    wcm, hcm = (nx + 2 * margin) * square_cm, (ny + 2 * margin) * square_cm
    tex = np.full((int(hcm * ppc), int(wcm * ppc)), 235, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x0 = int((margin + i) * square_cm * ppc)
                y0 = int((margin + j) * square_cm * ppc)
                tex[y0:int(y0 + square_cm * ppc), x0:int(x0 + square_cm * ppc)] = 25
    tex = cv2.GaussianBlur(tex, (0, 0), 0.6 * ppc / 10)
    # repère du damier : origine au 1er coin intérieur = coin (margin+1, margin+1) en cases
    origin = (margin + 1) * square_cm
    return tex, origin


def render_board(K, dist, rvec, tvec, pattern=(9, 6), square_cm=2.5, noise=2.0, blur=0.8,
                 rng=None, size=(W, H), ppc=40.0) -> np.ndarray:
    """Image en niveaux de gris du damier vu par la caméra (K, dist), pose (rvec, tvec)."""
    rng = rng or np.random.default_rng(0)
    tex, origin = _texture(pattern, square_cm, ppc)
    w, h = size
    grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)), -1)
    und = cv2.undistortPoints(grid.reshape(-1, 1, 2), K, dist,
                              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-6)
                              ).reshape(-1, 2)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    M = np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, np.float64).reshape(3)])
    P = np.linalg.inv(M) @ np.vstack([und.T, np.ones(len(und))])
    X, Y = P[0] / P[2], P[1] / P[2]
    mapx = ((X + origin) * ppc).reshape(h, w).astype(np.float32)
    mapy = ((Y + origin) * ppc).reshape(h, w).astype(np.float32)
    img = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=120)
    img = img.astype(np.float32)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def random_board_poses(n, rng, pattern=(9, 6), square_cm=2.5, z_range=(35.0, 60.0)) -> List[Tuple]:
    """Poses (rvec, tvec) du damier : inclinaisons jusqu'à ±30°, positions variées."""
    cols, rows = pattern
    cx_b, cy_b = (cols - 1) * square_cm / 2, (rows - 1) * square_cm / 2
    poses = []
    for _ in range(n):
        ang = np.radians(rng.uniform(-30, 30, 3) * np.array([1, 1, 0.5]))
        R, _ = cv2.Rodrigues(ang)
        z = rng.uniform(*z_range)
        centre = np.array([rng.uniform(-0.25, 0.25) * z * 0.5, rng.uniform(-0.2, 0.2) * z * 0.5, z])
        t = centre - R @ np.array([cx_b, cy_b, 0.0])     # le centre du damier en `centre`
        poses.append((cv2.Rodrigues(R)[0].reshape(3), t))
    return poses


def stereo_pair(baseline_cm: float = 10.0, toe_in_deg: float = 3.0):
    """Caméra 2 à droite de la caméra 1, légèrement tournée vers elle : x2 = R·x1 + T."""
    R, _ = cv2.Rodrigues(np.array([0.0, -np.radians(toe_in_deg), 0.0]))
    C2 = np.array([baseline_cm, 0.0, 0.0])               # centre de la caméra 2 (repère cam 1)
    T = -R @ C2
    return R, T


def project(K, R, T, X) -> np.ndarray:
    """Projection sans distorsion d'un point 3D (repère monde = caméra 1)."""
    x = K @ (R @ np.asarray(X, np.float64) + T)
    return x[:2] / x[2]


def letter_sheet_homography(rng, landscape=True):
    """Homographie feuille (cm) → image d'une feuille lettre vue en perspective."""
    w, h = (27.94, 21.59) if landscape else (21.59, 27.94)
    sheet = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    cx, cy = rng.uniform(250, 390), rng.uniform(200, 280)
    sx = rng.uniform(260, 340) / 2
    sy = sx * h / w * rng.uniform(0.6, 0.9)              # feuille vue de biais
    k = rng.uniform(0.75, 0.95)                          # trapèze (perspective)
    img = np.array([[cx - sx * k, cy - sy], [cx + sx * k, cy - sy], [cx + sx, cy + sy], [cx - sx, cy + sy]])
    img += rng.normal(0, 6, img.shape)
    return sheet, img, (w, h)
