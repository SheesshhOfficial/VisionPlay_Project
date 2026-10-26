"""Pure game logic and reproducible landmark feature extraction."""
import numpy as np
LABELS = ('rock', 'paper', 'scissors')
GESTURES = {'Closed_Fist': 'rock', 'Open_Palm': 'paper', 'Victory': 'scissors'}

def features(points):
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError('Expected 21 finite xyz landmarks')
    p = p - p[0]
    scale = np.linalg.norm(p, axis=1).max()
    if scale < 1e-8:
        raise ValueError('Degenerate landmarks')
    return (p / scale).ravel()

def outcome(player, computer):
    if player not in LABELS or computer not in LABELS:
        raise ValueError('Unknown move')
    if player == computer:
        return 'draw'
    return 'win' if (LABELS.index(player)-LABELS.index(computer)) % 3 == 1 else 'lose'
