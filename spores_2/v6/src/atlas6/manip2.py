"""Манипулятор 2 звена, кинематика q̇ = u, |u_i| ≤ 1, состояние — тор T² (ступень 1, research/manipulator_plan.md). 8 слоёв: 4 угла + 4 середины рёбер.
V в узлах тора (шаг h), выход клетки — билинейно (периодично); цель — |Δq_i| < Rg (по L∞). Точный эталон: T* = max_i |Δq_i mod 2π| (суставы независимы)."""
import numpy as np

LAYERS2 = [(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0) if (a, b) != (0.0, 0.0)]
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def T_star2(q):
    q = np.asarray(q, float); return np.max(np.abs(wrap(q)), axis=-1)


def solve_manip2(h=2 * np.pi / 48, tau=None, Rg=None, iters=5000, tol=1e-10):
    n = int(round(2 * np.pi / h)); h = 2 * np.pi / n; tau = h if tau is None else tau; Rg = h * 0.5 if Rg is None else Rg
    g = -np.pi + h * np.arange(n); Q1, Q2 = np.meshgrid(g, g, indexing='ij')
    goal = (np.abs(wrap(Q1)) < Rg) & (np.abs(wrap(Q2)) < Rg)
    V = np.full(Q1.shape, 4 * np.pi); V[goal] = 0.0

    def interp(V, a, b):
        f1 = ((a + np.pi) / h) % n; f2 = ((b + np.pi) / h) % n; i, j = np.floor(f1).astype(int), np.floor(f2).astype(int); u, w = f1 - i, f2 - j
        i2, j2 = (i + 1) % n, (j + 1) % n
        return (1 - u) * (1 - w) * V[i, j] + u * (1 - w) * V[i2, j] + (1 - u) * w * V[i, j2] + u * w * V[i2, j2]
    for it in range(iters):
        Vn = V.copy()
        for u1, u2 in LAYERS2:
            Vn = np.minimum(Vn, tau + interp(V, Q1 + u1 * tau, Q2 + u2 * tau))
        Vn[goal] = 0.0; ch = float(np.max(V - Vn)); V = Vn
        if ch < tol:
            break
    return dict(g=g, V=V, h=h, n=n, tau=tau, Rg=Rg, iters=it + 1)


def V_at2(A, q):
    h, n, V = A['h'], A['n'], A['V']; f = ((np.asarray(q, float) + np.pi) / h) % n; i, j = int(f[0]), int(f[1]); u, w = f[0] - i, f[1] - j
    return (1 - u) * (1 - w) * V[i, j] + u * (1 - w) * V[(i + 1) % n, j] + (1 - u) * w * V[i, (j + 1) % n] + u * w * V[(i + 1) % n, (j + 1) % n]
