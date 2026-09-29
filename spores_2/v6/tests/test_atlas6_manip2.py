"""Манипулятор 2 звена, кинематика: V по графу против точного T* = max|Δq_i| (с поправкой на допуск цели Rg)."""
import sys; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.manip2 import solve_manip2, V_at2, T_star2
E = []
for n in (24, 48):
    A = solve_manip2(h=2 * np.pi / n); rng = np.random.default_rng(0); P = rng.uniform(-np.pi, np.pi, (300, 2))
    d = np.array([V_at2(A, p) for p in P]) - np.maximum(T_star2(P) - A['Rg'], 0)
    E.append(abs(d).max()); print('n=%d iters %d: V−(T*−Rg): mean %+.4f max|d| %.4f' % (n, A['iters'], d.mean(), abs(d).max()))
assert E[1] < 0.7 * E[0], E                                  # ошибка убывает с h (порядок ≈ 1)
