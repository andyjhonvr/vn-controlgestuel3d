"""Régression : une tache de visage séparée du bord ne doit pas devenir une main.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g3d.gestures import GestureEngine, Sample
from g3d.hand import HandSegmenter, extract_hand, hand_protect_mask


SHAPE = (480, 640)


def face_mask(center=(320, 115)):
    mask = np.zeros(SHAPE, np.uint8)
    cv2.ellipse(mask, center, (55, 75), 0, 0, 360, 255, -1)
    return mask


def hand_mask(side):
    mask = np.zeros(SHAPE, np.uint8)
    center, end = {
        'bas': ((170, 310), (170, 479)),
        'gauche': ((150, 320), (0, 320)),
        'droite': ((490, 320), (639, 320)),
    }[side]
    cv2.line(mask, center, end, 255, 46)
    cv2.circle(mask, center, 38, 255, -1)
    return mask


def evaluate():
    rng = np.random.default_rng(42)
    counts = dict(face_selected=0, face_after_loss=0, hand_kept=0,
                  hand_removed=0, floating_rejected=0)
    for i in range(120):
        x, y = int(rng.integers(245, 395)), int(rng.integers(95, 150))
        face = face_mask((x, y))
        counts['face_selected'] += extract_hand(face) is not None
        # Ancienne position de main près du visage : le suivi ne doit pas s'y accrocher.
        counts['face_after_loss'] += extract_hand(face, prev_point=(x, y + 85)) is not None
        hand = hand_mask(('bas', 'gauche', 'droite')[i % 3])
        reference = extract_hand(hand)
        assert reference is not None
        together = extract_hand(cv2.bitwise_or(hand, face), prev_point=reference.point)
        counts['hand_kept'] += together is not None and together.point == reference.point
        counts['hand_removed'] += extract_hand(face, prev_point=reference.point) is None
        # Une coupure du bras (manche/fond) doit suspendre la commande.
        floating = hand.copy()
        floating[-3:, :] = floating[:, :3] = floating[:, -3:] = 0
        counts['floating_rejected'] += extract_hand(floating, prev_point=reference.point) is None
    print('Sur 120 scenes synthetiques :')
    for name, value in counts.items():
        print(f'  {name}: {value}/120')
    return counts


def test_moving_face_pipeline():
    """Couleur + soustraction du fond + choix de la main + gestes, sans webcam."""
    skin = cv2.cvtColor(np.uint8([[[170, 150, 105]]]), cv2.COLOR_YCrCb2BGR)[0, 0]
    background = np.full((*SHAPE, 3), 35, np.uint8)
    seg = HandSegmenter()
    engine = GestureEngine(640, volume_enabled=False)
    previous = None
    observations = events = foreground_frames = 0
    for i in range(120):
        frame = background.copy()
        if i >= 30:
            # Le visage bouge apres l'apprentissage, donc reapparait dans le masque brut.
            x = int(320 + 95 * np.sin((i - 30) / 9))
            cv2.ellipse(frame, (x, 125), (55, 75), 0, 0, 360,
                        tuple(int(v) for v in skin), -1)
        mask = seg.segment(frame)
        foreground_frames += bool(np.any(mask))
        obs = extract_hand(mask, prev_point=previous)
        previous = obs.point if obs is not None else None
        seg.update_background(hand_protect_mask(mask.shape, obs))
        sample = None if obs is None else Sample(i / 30, *obs.point, obs.fingers)
        observations += obs is not None
        events += engine.update(sample) is not None
    print(f'Visage mobile : {foreground_frames} images de masque non vide, '
          f'{observations} mains selectionnees, {events} commandes')
    assert foreground_frames > 0, 'La simulation doit produire un visage dans le masque'
    assert observations == 0 and events == 0


def test_tracking_limits():
    # Une tache au bord haut uniquement n'est pas une entree de bras autorisee.
    assert extract_hand(face_mask((320, 50))) is None
    left = extract_hand(hand_mask('gauche'))
    assert left is not None
    # Un nouveau contour eloigne ne doit pas fabriquer une trajectoire de balayage.
    assert extract_hand(hand_mask('droite'), prev_point=left.point) is None
    assert extract_hand(hand_mask('droite')) is not None
    print('OK : bord haut rejete, saut de suivi rejete puis nouvelle acquisition possible')


if __name__ == '__main__':
    counts = evaluate()
    assert counts['face_selected'] == counts['face_after_loss'] == 0
    assert counts['hand_kept'] == counts['hand_removed'] == counts['floating_rejected'] == 120
    test_moving_face_pipeline()
    test_tracking_limits()
    print('OK')
