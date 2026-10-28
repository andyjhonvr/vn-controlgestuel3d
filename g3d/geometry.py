"""Géométrie projective écrite à la main (cours, sem. 3-5 et 11-13).
"""
from __future__ import annotations

import numpy as np


def _normalize(pts: np.ndarray):
    """Centre les points et met leur distance moyenne à √2 (conditionnement de la DLT)."""
    pts = np.asarray(pts, np.float64)
    c = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - c, axis=1))
    s = np.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph, T


def dlt_homography(src, dst) -> np.ndarray:
    """H (3×3) telle que dst ~ H · src, à partir de n ≥ 4 correspondances (moindres carrés)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("au moins 4 correspondances sont nécessaires")
    a, Ta = _normalize(src)
    b, Tb = _normalize(dst)
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """Applique H à des points (N, 2) ou (2,); renvoie la même forme."""
    p = np.asarray(pts, np.float64)
    one = p.ndim == 1
    p = np.atleast_2d(p)
    q = np.c_[p, np.ones(len(p))] @ H.T
    q = q[:, :2] / q[:, 2:3]
    return q[0] if one else q


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1, x2) -> np.ndarray:
    """Point 3D (3,) dont les projections par P1 et P2 sont x1 et x2 (DLT, SVD)."""
    x1, x2 = np.asarray(x1, np.float64), np.asarray(x2, np.float64)
    A = np.array([x1[0] * P1[2] - P1[0],
                  x1[1] * P1[2] - P1[1],
                  x2[0] * P2[2] - P2[0],
                  x2[1] * P2[2] - P2[1]])
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]
