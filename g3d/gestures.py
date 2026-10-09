"""Moteur de gestes : transforme une suite d'échantillons de la main en commandes.

Ordre fixe des règles à chaque échantillon (voir CLAUDE.md, section 3) :
    1. aucune main ou main hors zone → reset();
    2. volume (commande continue, sans délai) : tant que le poing est fermé;
    3. délai (cooldown) de 1 s après un événement discret;
    4. balayage gauche / droite → ±10 s;
    5. main ouverte immobile → lecture / pause (avec un verrou simple).

Itérations 3 et 4. L'anti-retour, le réarmement complet et le journal CSV arrivent
à l'itération 5. Le moteur ne lit que des échantillons horodatés : il se teste sur
des trajectoires synthétiques, sans caméra.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from .config import GestureConfig


@dataclass
class Sample:
    t: float                         # temps (s)
    x: float                         # centre de paume dans l'image BRUTE (px)
    y: float
    fingers: int                     # nombre de doigts tendus
    z_cm: Optional[float] = None     # profondeur de la main (cm), None si inconnue
    in_zone: bool = True             # zone de contrôle (itération 6); toujours vrai d'ici là


@dataclass
class Event:
    kind: str                        # "seek", "play_pause" ou "volume"
    t: float
    value: Optional[float] = None    # seek : ±secondes; volume : nouveau volume (%)
    reason: str = ""                 # raison chiffrée, affichée et journalisée

    def __str__(self) -> str:
        return f"[{self.t:8.2f} s] {self.kind:<10} {self.reason}"


class GestureEngine:
    def __init__(self, frame_width: int, cfg: GestureConfig | None = None):
        self.cfg = cfg or GestureConfig()
        self.width = frame_width
        self.volume = self.cfg.volume_initial_pct
        self.cooldown_until = -np.inf
        self.history: Deque[Sample] = deque()
        # Volume (poing = embrayage)
        self.gripping = False
        self._z_buf: Deque[float] = deque(maxlen=self.cfg.volume_median_n)
        self._z0: Optional[float] = None
        self._v0 = self.volume
        self._miss = 0
        self.z_filtered: Optional[float] = None     # dernière médiane (affichage)
        # Verrou de la pause
        self._pause_latched = False
        self._latch_pos = (0.0, 0.0)

    # ------------------------------------------------------------------ état
    def reset(self) -> None:
        """Main absente ou hors zone : on oublie la trajectoire et on lâche le poing.
        Le volume courant et le délai sont conservés."""
        self.history.clear()
        self._release()
        self._pause_latched = False

    def _release(self) -> None:
        self.gripping = False
        self._z_buf.clear()
        self._z0 = None
        self._miss = 0
        self.z_filtered = None

    # ---------------------------------------------------------------- boucle
    def update(self, s: Optional[Sample]) -> Optional[Event]:
        # 1. aucune main ou main hors zone
        if s is None or not s.in_zone:
            self.reset()
            return None

        if self._pause_latched:
            self._update_latch(s)                         # le poing réarme aussi la pause

        # 2. volume : prioritaire et sans délai; pendant la prise, rien d'autre
        ev = self._volume(s)
        if self.gripping:
            self.history.clear()
            return ev

        self._push(s)

        # 3. délai après un événement discret
        if s.t < self.cooldown_until:
            return None
        # 4. balayage
        ev = self._swipe(s)
        if ev is None:
            # 5. main ouverte immobile
            ev = self._pause(s)
        if ev is not None:
            self.cooldown_until = s.t + self.cfg.cooldown_s
            self.history.clear()
        return ev

    def _push(self, s: Sample) -> None:
        self.history.append(s)
        keep = max(self.cfg.swipe_max_duration_s, self.cfg.still_duration_s) + 0.1
        # on garde un échantillon plus ancien que la fenêtre pour savoir qu'elle est couverte
        while len(self.history) > 1 and self.history[1].t <= s.t - keep:
            self.history.popleft()

    # ---------------------------------------------------------------- volume
    def _volume(self, s: Sample) -> Optional[Event]:
        c = self.cfg
        fist = s.fingers == 0
        if not self.gripping:
            if not fist:
                return None
            self.gripping, self._v0, self._miss = True, self.volume, 0
        elif not fist:
            # tolérance : quelques images mal comptées ne lâchent pas le poing
            self._miss += 1
            if self._miss > c.fist_tolerance_frames:
                self._release()
            return None
        self._miss = 0
        if s.z_cm is None:
            return None
        self._z_buf.append(s.z_cm)
        if len(self._z_buf) < c.volume_median_n:
            return None                                   # on remplit la médiane
        z = float(np.median(self._z_buf))
        self.z_filtered = z
        if self._z0 is None:
            self._z0 = z                                  # référence à la prise du poing
            return None
        dz = self._z0 - z                                 # > 0 : la main s'approche
        # Zone morte CONTINUE : on retranche 1 cm. Un seuil sec ferait sauter la cible de
        # 0 à ±2 % (= l'hystérésis) au premier dépassement, et le bruit suffirait.
        dz_eff = float(np.sign(dz) * max(0.0, abs(dz) - c.volume_dead_zone_cm))
        target = self._v0 + c.volume_gain_pct_per_cm * dz_eff
        target = float(np.clip(target, 0.0, 100.0))
        if abs(target - self.volume) < c.volume_hysteresis_pct:
            return None
        self.volume = target
        return Event("volume", s.t, target,
                     f"poing, Z0={self._z0:.1f} cm, Z={z:.1f} cm (médiane de {c.volume_median_n}), "
                     f"Z0-Z={dz:+.1f} cm (zone morte {c.volume_dead_zone_cm:.0f} cm) → volume {self._v0:.0f} % "
                     f"{c.volume_gain_pct_per_cm * dz_eff:+.0f} % "
                     f"= {target:.0f} %")

    # -------------------------------------------------------------- balayage
    def _swipe(self, s: Sample) -> Optional[Event]:
        c = self.cfg
        min_dx = c.swipe_min_dx_ratio * self.width
        for old in self.history:                          # du plus ancien au plus récent
            dt = s.t - old.t
            if dt > c.swipe_max_duration_s or dt <= 0:
                continue
            dx, dy = s.x - old.x, s.y - old.y
            if abs(dx) >= min_dx and abs(dy) <= c.swipe_max_dy_ratio * abs(dx):
                screen_dx = -dx if c.mirror else dx       # sens vu par l'utilisateur
                step = c.seek_step_s if screen_dx > 0 else -c.seek_step_s
                side = "droite" if screen_dx > 0 else "gauche"
                return Event("seek", s.t, step,
                             f"balayage {side} : |dx|={abs(dx):.0f} px ({100 * abs(dx) / self.width:.0f} % "
                             f"≥ {100 * c.swipe_min_dx_ratio:.0f} %) en {1000 * dt:.0f} ms, "
                             f"|dy|={abs(dy):.0f} px → {step:+.0f} s")
        return None

    # ----------------------------------------------------------------- pause
    def _update_latch(self, s: Sample) -> None:
        moved = np.hypot(s.x - self._latch_pos[0], s.y - self._latch_pos[1])
        if s.fingers == 0 or moved > self.cfg.pause_rearm_px:
            self._pause_latched = False

    def _pause(self, s: Sample) -> Optional[Event]:
        c = self.cfg
        if self._pause_latched or s.fingers < c.open_min_fingers:
            return None
        t0 = s.t - c.still_duration_s
        if self.history[0].t > t0:
            return None                                   # fenêtre pas encore couverte
        win: List[Sample] = [h for h in self.history if h.t >= t0]
        if any(h.fingers < c.open_min_fingers for h in win):
            return None
        pts = np.array([(h.x, h.y) for h in win])
        disp = float(np.max(np.hypot(*(pts - np.median(pts, axis=0)).T)))
        if disp > c.still_max_px:
            return None
        self._pause_latched = True
        self._latch_pos = (s.x, s.y)
        return Event("play_pause", s.t, None,
                     f"main ouverte ({s.fingers} doigts ≥ {c.open_min_fingers}) immobile "
                     f"{1000 * (s.t - win[0].t):.0f} ms, déplacement max {disp:.1f} px ≤ {c.still_max_px:.0f} px")
