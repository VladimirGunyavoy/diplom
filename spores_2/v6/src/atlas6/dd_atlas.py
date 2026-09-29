"""Атлас v6 для дифдрайва (x, y, θ), U = ромб |v|+|ω| ≤ 1: 4 слоя (прямая ±, поворот на месте ±). V в спорах сетки (h по x,y; hθ по θ, θ периодично).
Q_k(c) = τ + V(flow_k(c, τ)), V в выходе — трилинейно по 8 спорам. Цель — клетка |xy| < R_goal, |θ| < Rθ: V = 0 (точная формула не известна — эталон отдельно)."""
import numpy as np
from .dd3 import flow

LAYERS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]


def solve_dd(h=0.25, lim=3.0, nth=None, tau=None, R_goal=None, Rth=None, iters=3000, tol=1e-9):
    nth = nth or 4 * int(round(np.pi / (2 * h))); hth = 2 * np.pi / nth
    tau = h / 2 if tau is None else tau
    R_goal = h if R_goal is None else R_goal; Rth = hth if Rth is None else Rth
    xs = np.arange(-lim, lim + 1e-9, h); n = len(xs); ths = -np.pi + hth * np.arange(nth)
    X, Y, TH = np.meshgrid(xs, xs, ths, indexing='ij')
    P = np.stack([X, Y, TH], -1)
    ends = [flow(P, vw, tau) for vw in LAYERS]
    dth = (TH + np.pi) % (2 * np.pi) - np.pi
    goal = (np.hypot(X, Y) < R_goal) & (np.abs(dth) < Rth)
    ub = lambda E: 2 * np.pi + np.hypot(E[..., 0], E[..., 1])                     # верхняя оценка: повернуть, проехать, повернуть
    V = ub(P); V[goal] = 0.0                                                     # итерация Беллмана идёт вниз от верхней оценки — без бесконечностей

    def interp(V, E):
        fx = (E[..., 0] + lim) / h; fy = (E[..., 1] + lim) / h; ft = (E[..., 2] + np.pi) / hth
        inside = (fx >= 0) & (fy >= 0) & (fx <= n - 1) & (fy <= n - 1)
        fx = np.clip(fx, 0, n - 1 - 1e-9); fy = np.clip(fy, 0, n - 1 - 1e-9)
        i, j = fx.astype(int), fy.astype(int); u, w = fx - i, fy - j
        ft = ft % nth; k = np.floor(ft).astype(int); s = ft - k; k2 = (k + 1) % nth
        out = 0.0
        for di, wi in ((0, 1 - u), (1, u)):
            for dj, wj in ((0, 1 - w), (1, w)):
                for kk, wk in ((k, 1 - s), (k2, s)):
                    out = out + wi * wj * wk * V[i + di, j + dj, kk]
        return np.where(inside, out, ub(E))

    for it in range(iters):
        Vn = V.copy()
        for E in ends:
            Vn = np.minimum(Vn, tau + interp(V, E))
        Vn[goal] = 0.0; ch = float(np.max(V - Vn)); V = Vn
        if ch < tol:
            break
    return dict(xs=xs, ths=ths, V=V, iters=it + 1, h=h, hth=hth, tau=tau)
