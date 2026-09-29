"""Аналитический эталон T* (АТЛАС §5); с vmax — с насыщением скорости (АТЛАС §7)."""
import numpy as np
from .coords import A


def _T_unsat(x, v, a):
    if x + v * abs(v) / (2 * a) > 0:
        return (v + 2 * np.sqrt(a * x + v * v / 2)) / a
    return (-v + 2 * np.sqrt(-a * x + v * v / 2)) / a


def T_star(x, v, a=A, vmax=None):
    if vmax is None:
        return _T_unsat(x, v, a)
    if x + v * abs(v) / (2 * a) < 0:      # зеркало: ниже кривой = выше для (−x, −v)
        return T_star(-x, -v, a, vmax)
    # выше кривой: сначала −a до v_m = −sqrt(a x + v²/2)
    if np.sqrt(a * x + v * v / 2) <= vmax:
        return _T_unsat(x, v, a)
    dist = x + v * v / (2 * a) - vmax ** 2 / a          # путь круиза влево
    return (v + vmax) / a + vmax / a + dist / vmax
