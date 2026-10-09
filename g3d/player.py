"""Lecteur vidéo piloté par les événements du moteur de gestes.

Deux sources :
  - une vraie vidéo (python3 main.py --video film.mp4), lue avec cv2.VideoCapture;
  - sinon une vidéo synthétique (horloge, barre de progression, disque qui tourne),
    pratique pour voir les sauts de ±10 s d'un coup d'œil.
OpenCV ne joue pas le son : le volume est affiché sous forme de barre.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .config import PlayerConfig
from .gestures import Event


def _fmt(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


class VideoPlayer:
    def __init__(self, path: Optional[str] = None, cfg: PlayerConfig | None = None,
                 volume: float = 50.0, clock=time.monotonic):
        self.cfg = cfg or PlayerConfig()
        self.clock = clock
        self.cap = None
        self.fps = 30.0
        self.duration = self.cfg.synthetic_duration_s
        if path is not None:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise SystemExit(f"Vidéo {path} illisible")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.duration = n / self.fps if n > 0 else self.duration
        self.playing = True
        self.volume = volume
        self._pos = 0.0                      # position (s) au moment _t_ref
        self._t_ref = self.clock()
        self._frame_idx = -1                 # dernière image lue (vraie vidéo)
        self._last_frame: Optional[np.ndarray] = None
        self._msg, self._msg_until = "", 0.0

    # ------------------------------------------------------------- position
    @property
    def position(self) -> float:
        p = self._pos + (self.clock() - self._t_ref if self.playing else 0.0)
        return float(np.clip(p, 0.0, self.duration))

    def _set_position(self, p: float) -> None:
        self._pos = float(np.clip(p, 0.0, self.duration))
        self._t_ref = self.clock()

    # ----------------------------------------------------------- commandes
    def toggle(self) -> None:
        self._set_position(self.position)
        self.playing = not self.playing

    def seek(self, delta_s: float) -> None:
        self._set_position(self.position + delta_s)

    def handle(self, ev: Optional[Event]) -> None:
        if ev is None:
            return
        if ev.kind == "seek":
            self.seek(ev.value)
            msg = f"{'>>' if ev.value > 0 else '<<'} {ev.value:+.0f} s"
        elif ev.kind == "play_pause":
            self.toggle()
            msg = "LECTURE" if self.playing else "PAUSE"
        elif ev.kind == "volume":
            self.volume = float(ev.value)
            msg = f"VOLUME {self.volume:.0f} %"
        else:
            return
        self._msg, self._msg_until = msg, self.clock() + self.cfg.message_s

    # ------------------------------------------------------------- images
    def _synthetic(self, p: float) -> np.ndarray:
        w, h = self.cfg.width, self.cfg.height
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (40 + int(p) % 60, 30, 60)               # la teinte change chaque seconde
        ang = 2 * np.pi * p / 10.0                         # un tour toutes les 10 s
        c = (w // 2 + int(110 * np.cos(ang)), h // 2 - 20 + int(110 * np.sin(ang)))
        cv2.circle(img, (w // 2, h // 2 - 20), 110, (90, 90, 90), 1)
        cv2.circle(img, c, 18, (0, 200, 255), -1)
        cv2.putText(img, _fmt(p), (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        return img

    def _real(self, p: float) -> np.ndarray:
        target = int(p * self.fps)
        if target != self._frame_idx:
            if target != self._frame_idx + 1:              # saut ou retard : on repositionne
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = self.cap.read()
            if ok:
                self._frame_idx = target
                self._last_frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        if self._last_frame is None:
            return np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
        return self._last_frame.copy()

    def render(self) -> np.ndarray:
        p = self.position
        img = self._real(p) if self.cap is not None else self._synthetic(p)
        h, w = img.shape[:2]
        # barre de progression
        cv2.rectangle(img, (10, h - 22), (w - 10, h - 12), (80, 80, 80), -1)
        cv2.rectangle(img, (10, h - 22), (10 + int((w - 20) * p / max(self.duration, 1e-6)), h - 12),
                      (0, 0, 255), -1)
        etat = "LECTURE" if self.playing else "PAUSE"
        cv2.putText(img, f"{etat}  {_fmt(p)} / {_fmt(self.duration)}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # barre de volume (verticale, à droite)
        top, bot = 20, h - 50
        cv2.rectangle(img, (w - 30, top), (w - 14, bot), (80, 80, 80), -1)
        cv2.rectangle(img, (w - 30, bot - int((bot - top) * self.volume / 100)), (w - 14, bot),
                      (0, 220, 0), -1)
        cv2.putText(img, f"{self.volume:.0f}%", (w - 70, top + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        # dernier événement
        if self.clock() < self._msg_until:
            cv2.putText(img, self._msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        return img

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
