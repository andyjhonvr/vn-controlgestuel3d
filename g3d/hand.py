"""Segmentation de la main et extraction de caractéristiques géométriques.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import FingerConfig, SegmentationConfig


@dataclass
class HandObservation:
    contour: np.ndarray
    area: float                                   # px²
    centroid: Tuple[float, float]                 # px
    palm_center: Tuple[float, float]              # px
    palm_radius: float                            # px
    bbox: Tuple[int, int, int, int]               # x, y, largeur, hauteur
    arm_entry: Optional[Tuple[float, float]] = None   # où le bras touche le bord de l'image
    fingers: int = 0                                  # nombre de doigts tendus (0 à 5)
    finger_valleys: List[Tuple[int, int]] = field(default_factory=list)  # fonds des creux (px)
    hull: Optional[np.ndarray] = None                 # enveloppe convexe (points)
    solidity: float = 1.0                             # aire / aire de l'enveloppe (poing ≈ 1)

    @property
    def point(self) -> Tuple[float, float]:
        """Point suivi pour les gestes : le centre de la paume, stable quand les doigts bougent."""
        return self.palm_center


class HandSegmenter:
    def __init__(self, cfg: SegmentationConfig | None = None):
        self.cfg = cfg or SegmentationConfig()
        self.background: Optional[np.ndarray] = None
        self.warmup_left = self.cfg.bg_warmup_frames
        self.last_skin: Optional[np.ndarray] = None   # masques intermédiaires (diagnostic)
        self.last_fg: Optional[np.ndarray] = None
        self._last_gray: Optional[np.ndarray] = None
        k = self.cfg.morph_kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ------------------------------------------------------------------ fond
    @property
    def warming_up(self) -> bool:
        return self.warmup_left > 0

    def reset_background(self) -> None:
        """Relance l'apprentissage du fond (touche b). La main doit être hors du champ."""
        self.warmup_left = self.cfg.bg_warmup_frames

    def _learn(self, gray: np.ndarray) -> None:
        self.background = gray.astype(np.float32)

    def update_background(self, protect: Optional[np.ndarray] = None) -> None:
        """Mise à jour lente du fond partout sauf dans `protect` (la main retenue).
        À appeler après extract_hand, une fois la main choisie."""
        if self.background is None or self._last_gray is None or self.warming_up:
            return
        keep = None if protect is None else cv2.bitwise_not(protect)
        cv2.accumulateWeighted(self._last_gray.astype(np.float32), self.background,
                               self.cfg.bg_alpha, mask=keep)

    def _foreground_mask(self, gray: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        fg = (diff > self.cfg.bg_diff_thresh).astype(np.uint8) * 255
        return cv2.dilate(fg, self._kernel, iterations=2)

    # ---------------------------------------------------------- segmentation
    def skin_mask(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        c = self.cfg
        return cv2.inRange(ycrcb, (0, c.cr_min, c.cb_min), (255, c.cr_max, c.cb_max))

    def segment(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._last_gray = gray
        mask = self.skin_mask(frame)
        self.last_skin, self.last_fg = mask, None
        if self.cfg.use_background:
            # Apprentissage : la webcam ajuste souvent son exposition pendant la première
            # seconde; on réapprend le fond à chaque image pendant ce temps.
            if self.warming_up or self.background is None:
                self._learn(gray)
                self.warmup_left = max(0, self.warmup_left - 1)
                return np.zeros_like(mask)
            fg = self._foreground_mask(gray)
            self.last_fg = fg
            mask = cv2.bitwise_and(mask, fg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return mask


# -------------------------------------------------------------- choix de la main
def _touches_arm_border(contour: np.ndarray, shape) -> bool:
    """Vrai si le contour touche le bord gauche, droit ou bas (entrée possible du bras)."""
    h, w = shape
    pts = contour[:, 0, :]
    return bool(np.any(pts[:, 0] <= 1) or np.any(pts[:, 0] >= w - 2) or np.any(pts[:, 1] >= h - 2))


def _select_contour(contours, shape, prev_point, cfg: SegmentationConfig):
    min_area = cfg.min_hand_area_ratio * shape[0] * shape[1]
    cands = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not cands:
        return None
    max_jump = cfg.track_max_jump_ratio * shape[1]

    def dist_to_prev(c):
        if prev_point is None:
            return 0.0
        # distance signée au contour : négative à l'extérieur
        return max(0.0, -cv2.pointPolygonTest(c, (float(prev_point[0]), float(prev_point[1])), True))

    arm = [c for c in cands if _touches_arm_border(c, shape)]
    if arm:                                   # 1. une tache par laquelle entre le bras
        if prev_point is not None:
            return min(arm, key=dist_to_prev)
        return max(arm, key=cv2.contourArea)
    if prev_point is not None:                # 2. suivi : la tache la plus proche de la main précédente
        best = min(cands, key=dist_to_prev)
        if dist_to_prev(best) <= max_jump:
            return best
    return max(cands, key=cv2.contourArea)    # 3. par défaut : la plus grande


# -------------------------------------------------------------- caractéristiques
def _palm_center(filled: np.ndarray, cfg: SegmentationConfig):
    """Centre et rayon de la paume.

    La transformée de distance donne, pour chaque pixel, la distance au bord de la
    silhouette; son maximum est le centre du plus grand disque inscrit. Mais si le bras
    entre dans l'image, l'avant-bras peut contenir un disque aussi grand que la paume.

    Solution géométrique : on repère l'entrée du bras (pixels de la silhouette sur le
    bord de l'image) et le bout de la main (point le plus éloigné de cette entrée).
    En partant du bout, on construit le profil « demi-largeur maximale en fonction de
    la distance au bout ». Il monte dans les doigts et la paume, redescend au poignet,
    puis remonte dans l'avant-bras. La paume est le premier sommet suivi d'une baisse
    d'au moins (1 - wrist_dip_ratio).
    """
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    border = np.zeros(filled.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    by, bx = np.nonzero((filled > 0) & border)
    _, r_global, _, c_global = cv2.minMaxLoc(dist)
    if len(bx) == 0:                                  # main « flottante » : maximum global
        return (float(c_global[0]), float(c_global[1])), float(r_global), None

    entry = (float(bx.mean()), float(by.mean()))
    ys, xs = np.nonzero(filled)
    d_entry = np.hypot(xs - entry[0], ys - entry[1])
    s = d_entry.max() - d_entry                       # distance depuis le bout de la main
    b = (s // cfg.profile_bin_px).astype(int)
    vals = dist[ys, xs]
    profile = np.zeros(b.max() + 1)
    np.maximum.at(profile, b, vals)                   # demi-largeur max par tranche

    peak_bin, peak = None, 0.0
    for i, p in enumerate(profile):
        if p > peak:
            peak, peak_bin = p, i
        elif peak >= cfg.palm_min_ratio * r_global and p < cfg.wrist_dip_ratio * peak:
            break                                     # poignet trouvé : la paume est le sommet
    else:
        return (float(c_global[0]), float(c_global[1])), float(r_global), entry

    sel = np.abs(b - peak_bin) <= 1
    k = int(np.argmax(np.where(sel, vals, -1)))
    return (float(xs[k]), float(ys[k])), float(vals[k]), entry


# -------------------------------------------------------------- doigts
def _angle_at(far, start, end) -> float:
    """Angle (degrés) au sommet `far` du triangle (start, far, end), par la loi des cosinus."""
    a = np.linalg.norm(np.subtract(start, end))
    b = np.linalg.norm(np.subtract(start, far))
    c = np.linalg.norm(np.subtract(end, far))
    if b * c == 0:
        return 180.0
    cos = np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _narrow_protrusions(contour: np.ndarray, center, r: float, cfg: FingerConfig) -> int:
    """Nombre de protubérances étroites (doigts isolés) hors d'un cercle de rayon ring·r.
    On parcourt le contour : chaque suite de points « dehors » est une protubérance;
    sa corde (distance entre son premier et son dernier point) mesure sa largeur."""
    pts = contour[:, 0, :].astype(np.float32)
    d = np.linalg.norm(pts - np.array(center, np.float32), axis=1)
    outside = d > cfg.protrusion_ring * r
    if outside.all() or not outside.any():
        return 0
    start = int(np.argmin(outside))            # un point intérieur : on déroule depuis là
    pts, d, outside = (np.roll(a, -start, axis=0) for a in (pts, d, outside))
    count, i, n = 0, 0, len(pts)
    while i < n:
        if outside[i]:
            j = i
            while j < n and outside[j]:
                j += 1
            chord = np.linalg.norm(pts[i] - pts[j - 1])
            if chord < cfg.protrusion_max_chord * r and d[i:j].max() > cfg.protrusion_min_reach * r:
                count += 1
            i = j
        else:
            i += 1
    return count


def count_fingers(contour: np.ndarray, palm_center, palm_radius: float,
                  cfg: FingerConfig | None = None) -> Tuple[int, List[Tuple[int, int]]]:
    """Nombre de doigts tendus et position des creux entre les doigts.

    Un défaut de convexité (start, end, far, profondeur) est un creux entre deux doigts si :
      - sa profondeur ≥ min_defect_depth_ratio × rayon de paume (indépendant de l'échelle);
      - l'angle au fond du creux ≤ max_defect_angle_deg (creux aigu);
      - ses deux bords (start, end) sont à ≥ tip_min_ratio × r du centre de paume :
        un creux contre une jointure ou un pouce replié est ignoré.
    Doigts = nombre de bouts distincts parmi les bords des creux retenus.
    Sans creux : 0 (poing) ou 1 (protubérance étroite).
    """
    cfg = cfg or FingerConfig()
    r = max(palm_radius, 1.0)
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return 0, []
    hull_idx = np.sort(hull_idx, axis=0)      # convexityDefects exige des indices croissants
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:                          # contour auto-intersecté (rare)
        defects = None

    center = np.asarray(palm_center, np.float64)
    valleys, tips = [], []
    if defects is not None and len(defects) > 0:
        # Selon la version d'OpenCV, la forme est (N, 1, 4) ou (N, 4) : on uniformise
        for s_i, e_i, f_i, depth in defects.reshape(-1, 4):
            start, end, far = contour[s_i][0], contour[e_i][0], contour[f_i][0]
            if depth / 256.0 < cfg.min_defect_depth_ratio * r:
                continue
            if _angle_at(far, start, end) > cfg.max_defect_angle_deg:
                continue
            # Les deux bords du creux doivent être des bouts de doigts tendus : un creux
            # contre une jointure ou un pouce replié ajouterait un doigt fantôme.
            if min(np.linalg.norm(start - center), np.linalg.norm(end - center)) < cfg.tip_min_ratio * r:
                continue
            valleys.append((int(far[0]), int(far[1])))
            tips.extend([start.astype(np.float64), end.astype(np.float64)])
    if valleys:
        # Doigts = bouts distincts (deux creux voisins partagent le bout du doigt du milieu)
        distinct = []
        for t in tips:
            if all(np.linalg.norm(t - u) >= cfg.tip_merge_ratio * r for u in distinct):
                distinct.append(t)
        return min(len(distinct), 5), valleys
    return (1 if _narrow_protrusions(contour, palm_center, r, cfg) > 0 else 0), []


def extract_hand(mask: np.ndarray, seg_cfg: SegmentationConfig | None = None,
                 prev_point: Optional[Tuple[float, float]] = None,
                 finger_cfg: FingerConfig | None = None) -> Optional[HandObservation]:
    seg_cfg = seg_cfg or SegmentationConfig()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = _select_contour(contours, mask.shape, prev_point, seg_cfg)
    if contour is None:
        return None
    area = cv2.contourArea(contour)

    m = cv2.moments(contour)
    centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    center, radius, entry = _palm_center(filled, seg_cfg)
    fingers, valleys = count_fingers(contour, center, radius, finger_cfg)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    return HandObservation(contour=contour, area=area, centroid=centroid,
                           palm_center=center, palm_radius=radius,
                           bbox=cv2.boundingRect(contour), arm_entry=entry,
                           fingers=fingers, finger_valleys=valleys, hull=hull,
                           solidity=area / hull_area if hull_area > 0 else 1.0)


def hand_protect_mask(shape, obs: Optional[HandObservation], kernel_px: int = 15) -> Optional[np.ndarray]:
    """Masque de la main à protéger lors de la mise à jour du fond. On ne protège que
    la main dont le bras entre par le bord : une tache « flottante » (souvent le visage)
    n'est pas protégée et finit absorbée par le fond si elle reste immobile."""
    if obs is None or obs.arm_entry is None:
        return None
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [obs.contour], -1, 255, cv2.FILLED)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.dilate(m, k)


def draw_hand(img: np.ndarray, obs: HandObservation) -> np.ndarray:
    out = img.copy()
    cv2.drawContours(out, [obs.contour], -1, (0, 255, 0), 2)
    if obs.hull is not None:
        cv2.drawContours(out, [obs.hull], -1, (255, 200, 0), 1)          # enveloppe convexe
    for v in obs.finger_valleys:
        cv2.circle(out, v, 6, (255, 0, 255), -1)                           # creux (magenta)
    cx, cy = map(int, obs.palm_center)
    cv2.circle(out, (cx, cy), int(obs.palm_radius), (0, 0, 255), 1)
    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    gx, gy = map(int, obs.centroid)
    cv2.drawMarker(out, (gx, gy), (255, 200, 0), cv2.MARKER_CROSS, 12, 2)
    if obs.arm_entry is not None:
        ex, ey = map(int, obs.arm_entry)
        cv2.circle(out, (ex, ey), 7, (0, 255, 255), -1)          # entrée du bras (jaune)
    return out