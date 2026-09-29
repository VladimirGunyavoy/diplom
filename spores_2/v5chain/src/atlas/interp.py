"""Запрос T из произвольной точки: билинейно в ячейке (c, d) (АТЛАС §4 п.6)."""
import numpy as np
from .coords import A, c_plus, c_minus
from .lattice import key


def interp_T(x, v, C, D, T, a=A):
    c, d, s = c_plus(x, v, a), c_minus(x, v, a), (1 if v >= 0 else -1)
    i = np.searchsorted(C, c) - 1
    j = np.searchsorted(D, d) - 1
    if i < 0 or j < 0 or i + 1 >= len(C) or j + 1 >= len(D):
        return np.nan
    corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
    vals = [T.get(key(ii, jj, s, C, D)) for ii, jj in corners]
    if any(t is None for t in vals):
        return np.nan
    u = (c - C[i]) / (C[i + 1] - C[i])
    w = (d - D[j]) / (D[j + 1] - D[j])
    return ((1-u)*(1-w)*vals[0] + u*(1-w)*vals[1]
            + (1-u)*w*vals[2] + u*w*vals[3])
