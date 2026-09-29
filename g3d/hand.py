"""Segmentation de la main et extraction de caractéristiques géométriques.

Itération 1 :
    1. masque de peau dans l'espace YCrCb (seuils fixes sur Cr et Cb);
    2. masque d'avant-plan par soustraction d'un fond appris (moyenne glissante);
    3. ET logique des deux masques, puis ouverture et fermeture morphologiques;
    4. choix de la main parmi les taches : celle par laquelle le bras entre dans
       l'image (bord gauche, droit ou bas), sinon la plus proche de la main précédente;
    5. centre de paume = maximum de la transformée de distance; si le bras entre
       dans l'image, on prend le premier élargissement avant le poignet, en partant
       du bout de la main (sinon l'avant-bras peut attirer le point).
Le fond est mis à jour partout sauf sur la main retenue : une tache parasite
immobile (le visage, par exemple) finit donc par être absorbée par le fond.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig


@dataclass
class HandObservation:
    contour: np.ndarray
    area: float                                   # px²
    centroid: Tuple[float, float]                 # px
    palm_center: Tuple[float, float]              # px
    palm_radius: float                            # px
    bbox: Tuple[int, int, int, int]               # x, y, largeur, hauteur
    arm_entry: Optional[Tuple[float, float]] = None   # où le bras touche le bord de l'image

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


def extract_hand(mask: np.ndarray, seg_cfg: SegmentationConfig | None = None,
                 prev_point: Optional[Tuple[float, float]] = None) -> Optional[HandObservation]:
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

    return HandObservation(contour=contour, area=area, centroid=centroid,
                           palm_center=center, palm_radius=radius,
                           bbox=cv2.boundingRect(contour), arm_entry=entry)


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
    cx, cy = map(int, obs.palm_center)
    cv2.circle(out, (cx, cy), int(obs.palm_radius), (0, 0, 255), 1)
    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    gx, gy = map(int, obs.centroid)
    cv2.drawMarker(out, (gx, gy), (255, 200, 0), cv2.MARKER_CROSS, 12, 2)
    if obs.arm_entry is not None:
        ex, ey = map(int, obs.arm_entry)
        cv2.circle(out, (ex, ey), 7, (0, 255, 255), -1)          # entrée du bras (jaune)
    return out