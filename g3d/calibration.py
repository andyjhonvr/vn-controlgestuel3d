"""Étalonnage des caméras par damier (itérations 7 et 8; cours, sem. 3-5 et 11-13).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def board_points(pattern: Tuple[int, int], square_cm: float) -> np.ndarray:
    """Coins du damier dans son repère (plan Z = 0), en cm : (N, 3) float32."""
    cols, rows = pattern
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_cm
    return g


def find_corners(img: np.ndarray, pattern: Tuple[int, int]) -> Optional[np.ndarray]:
    """Coins intérieurs (N, 1, 2) float32 raffinés au sous-pixel, ou None."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, tuple(pattern), flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX)
    return corners.reshape(-1, 1, 2).astype(np.float32)


def calibrate_mono(image_points: Sequence[np.ndarray], image_size: Tuple[int, int],
                   pattern: Tuple[int, int], square_cm: float) -> Dict:
    """K, dist, rms (px) et erreur par vue. image_size = (largeur, hauteur)."""
    obj = board_points(pattern, square_cm)
    objs = [obj] * len(image_points)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, list(image_points), tuple(image_size), None, None)
    per_view = []
    for o, ip, rv, tv in zip(objs, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))))
    return {"K": K, "dist": dist, "rms": float(rms), "size": tuple(image_size),
            "per_view": per_view, "rvecs": rvecs, "tvecs": tvecs}


def calibrate_stereo(pts1: Sequence[np.ndarray], pts2: Sequence[np.ndarray], image_size: Tuple[int, int],
                     pattern: Tuple[int, int], square_cm: float) -> Dict:
    """Étalonne les deux caméras puis leur position relative (x2 = R·x1 + T)."""
    m1 = calibrate_mono(pts1, image_size, pattern, square_cm)
    m2 = calibrate_mono(pts2, image_size, pattern, square_cm)
    obj = board_points(pattern, square_cm)
    rms, K1, d1, K2, d2, R, T, _, _ = cv2.stereoCalibrate(
        [obj] * len(pts1), list(pts1), list(pts2), m1["K"], m1["dist"], m2["K"], m2["dist"],
        tuple(image_size), flags=cv2.CALIB_FIX_INTRINSIC, criteria=_SUBPIX)
    return {"K1": K1, "d1": d1, "K2": K2, "d2": d2, "R": R, "T": T.reshape(3), "rms": float(rms),
            "rms1": m1["rms"], "rms2": m2["rms"], "size": tuple(image_size)}


# ------------------------------------------------------------------ fichiers
def save_mono(path: str, calib: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, K=calib["K"], dist=calib["dist"], size=np.array(calib["size"]), rms=calib["rms"])


def load_mono(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"K": d["K"], "dist": d["dist"], "size": tuple(int(v) for v in d["size"]), "rms": float(d["rms"])}


def save_stereo(path: str, calib: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **{k: np.asarray(calib[k]) for k in ("K1", "d1", "K2", "d2", "R", "T", "rms", "size")})


def load_stereo(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    d = np.load(path)
    out = {k: d[k] for k in ("K1", "d1", "K2", "d2", "R", "T")}
    out["rms"] = float(d["rms"])
    out["size"] = tuple(int(v) for v in d["size"])
    return out


def focal_for_frame(K: np.ndarray, calib_size: Tuple[int, int], frame_width: int) -> float:
    """fx (px) mise à l'échelle si l'image courante n'a pas la largeur de l'étalonnage."""
    return float(K[0, 0]) * frame_width / float(calib_size[0])


def corners_from_folder(folder: str, pattern: Tuple[int, int], prefix: str = "") -> Tuple[List[str], List[np.ndarray], Optional[Tuple[int, int]]]:
    """Coins trouvés dans les images d'un dossier (triées par nom)."""
    names, pts, size = [], [], None
    if not os.path.isdir(folder):
        return names, pts, size
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".png", ".jpg")) or not f.startswith(prefix):
            continue
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        c = find_corners(img, pattern)
        if c is not None:
            names.append(f)
            pts.append(c)
            size = (img.shape[1], img.shape[0])
    return names, pts, size
