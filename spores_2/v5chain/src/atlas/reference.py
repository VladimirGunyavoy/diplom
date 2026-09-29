"""Аналитический эталон T* (АТЛАС §5)."""
import numpy as np
from .coords import A


def T_star(x, v, a=A):
    if x + v * abs(v) / (2 * a) > 0:
        return (v + 2 * np.sqrt(a * x + v * v / 2)) / a
    return (-v + 2 * np.sqrt(-a * x + v * v / 2)) / a
