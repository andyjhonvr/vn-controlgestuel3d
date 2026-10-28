"""Moteur de gestes : transforme une suite d'échantillons de la main en commandes.
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

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
    kind: str                        # "seek", "play_pause", "volume" (ou "seek_bloque", journal)
    t: float
    value: Optional[float] = None    # seek : ±secondes; volume : nouveau volume (%)
    reason: str = ""                 # raison chiffrée, affichée et journalisée

    def __str__(self) -> str:
        return f"[{self.t:8.2f} s] {self.kind:<10} {self.reason}"


class GestureEngine:
    def __init__(self, frame_width: int, cfg: GestureConfig | None = None,
                 *, volume_enabled: bool = True, robust: bool = True):
        self.cfg = cfg or GestureConfig()
        self.volume_enabled = volume_enabled
        # robust=False reproduit exactement les itérations 3 et 4 (démonstrations)
        self.robust = robust
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
        # Verrou de la pause : "pause" (poing ou déplacement) ou "balayage" (poing seulement)
        self._pause_latched = False
        self._latch_kind = "pause"
        self._latch_pos = (0.0, 0.0)
        # Itération 5 : anti-retour et tolérance aux images sans main
        self._last_swipe: Optional[Tuple[float, float, float, float]] = None  # t, signe dx brut, x départ, x arrivée
        self._missing = 0
        self.suppressed: Deque[Event] = deque(maxlen=100)   # balayages bloqués (journal)

    # ------------------------------------------------------------------ état
    def reset(self) -> None:
        """Main absente ou hors zone : on oublie la trajectoire et on lâche le poing.
        Le volume courant et le délai sont conservés."""
        self.history.clear()
        self._release()
        self._pause_latched = False
        self._last_swipe = None                           # sortir de la zone réarme les deux sens
        self._missing = 0

    def _release(self) -> None:
        self.gripping = False
        self._z_buf.clear()
        self._z0 = None
        self._miss = 0
        self.z_filtered = None

    # ---------------------------------------------------------------- boucle
    def update(self, s: Optional[Sample]) -> Optional[Event]:
        # 1. aucune main ou main hors zone
        if s is None and self.robust and self._missing < self.cfg.missing_tolerance_frames:
            self._missing += 1                            # image manquée tolérée : on attend
            return None
        if s is None or not s.in_zone:
            self.reset()
            return None
        self._missing = 0

        if self._pause_latched:
            self._update_latch(s)                         # le poing réarme aussi la pause

        # 2. volume : prioritaire et sans délai; pendant la prise, rien d'autre
        if self.volume_enabled:
            ev = self._volume(s)
            if self.gripping:
                self.history.clear()
                return ev
        elif s.fingers == 0:
            # Itération 3 : le poing réarme la pause, sans commande ni trajectoire.
            self.history.clear()
            return None

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
            if ev.kind == "seek" and self.robust:
                # la main ouverte immobile en fin de balayage ne doit pas mettre en pause
                self._pause_latched, self._latch_kind = True, "balayage"
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
                if self.robust:
                    why = self._antireturn(s.t, old.x, dx)
                    if why:
                        self.suppressed.append(Event("seek_bloque", s.t, step,
                                                     f"balayage {side} bloqué : {why}"))
                        self.history.clear()              # ce retour est consommé
                        return None
                    self._last_swipe = (s.t, float(np.sign(dx)), old.x, s.x)
                return Event("seek", s.t, step,
                             f"balayage {side} : |dx|={abs(dx):.0f} px ({100 * abs(dx) / self.width:.0f} % "
                             f"≥ {100 * c.swipe_min_dx_ratio:.0f} %) en {1000 * dt:.0f} ms, "
                             f"|dy|={abs(dy):.0f} px → {step:+.0f} s")
        return None

    def _antireturn(self, t: float, x_start: float, dx: float) -> str:
        """Raison du blocage si ce balayage est le retour naturel du précédent, sinon ""."""
        if self._last_swipe is None:
            return ""
        t_prev, sgn, xa, xb = self._last_swipe
        if t - t_prev > self.cfg.antireturn_s or np.sign(dx) == sgn:
            return ""
        if (x_start - xa) * sgn <= 0:
            return ""                                     # part de l'autre côté du départ précédent
        return (f"anti-retour, {1000 * (t - t_prev):.0f} ms ≤ {1000 * self.cfg.antireturn_s:.0f} ms "
                f"après le balayage précédent ({xa:.0f} → {xb:.0f} px), départ à {x_start:.0f} px "
                f"du côté de son arrivée")

    # ----------------------------------------------------------------- pause
    def _update_latch(self, s: Sample) -> None:
        if s.fingers == 0:
            self._pause_latched = False
            return
        moved = np.hypot(s.x - self._latch_pos[0], s.y - self._latch_pos[1])
        if self._latch_kind == "pause" and moved > self.cfg.pause_rearm_px:
            self._pause_latched = False

    def _pause(self, s: Sample) -> Optional[Event]:
        c = self.cfg
        if self._pause_latched or s.fingers < c.open_min_fingers or not self.history:
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
        self._pause_latched, self._latch_kind = True, "pause"
        self._latch_pos = (s.x, s.y)
        return Event("play_pause", s.t, None,
                     f"main ouverte ({s.fingers} doigts ≥ {c.open_min_fingers}) immobile "
                     f"{1000 * (s.t - win[0].t):.0f} ms, déplacement max {disp:.1f} px ≤ {c.still_max_px:.0f} px")


class EventLogger:
    """Journal CSV des événements et de leur raison chiffrée (results/events_log.csv).
    Le fichier est complété à chaque lancement; la colonne `session` les distingue."""

    HEADER = ["session", "t_s", "type", "valeur", "raison"]

    def __init__(self, path: str = os.path.join("results", "events_log.csv")):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.path = path
        self.session = time.strftime("%Y-%m-%d %H:%M:%S")
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(self.HEADER)

    def log(self, ev: Event) -> None:
        value = "" if ev.value is None else f"{ev.value:g}"
        self._w.writerow([self.session, f"{ev.t:.3f}", ev.kind, value, ev.reason])
        self._f.flush()

    def close(self) -> None:
        self._f.close()
