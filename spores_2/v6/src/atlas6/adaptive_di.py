"""Адаптивные споры DI (эксперимент H1/H2, research/adaptive_vs_grid.md): споры под запрос (старт → цель) вместо равномерной решётки.
Споры: цепочки вдоль потока от старта (оба слоя, шаг τ) + обратные цепочки от цели (кривая переключения) + «веера» — цепочки другого слоя от каждой m-й споры первой цепочки.
V — итерации Беллмана по графу «спора → выход потока на τ», значение в выходе — линейная (Делоне) интерполяция по спорам, старт итераций K (сверху)."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from .cell import flow
from .agent import T_star


def spores_for(x0, tau, T_b, m=1, Rg=0.25):
    K = int(np.ceil(T_b / tau)); P = [np.zeros(2), np.asarray(x0, float)]
    for s in (+1.0, -1.0):
        fwd = [flow(np.asarray(x0, float), s, k * tau) for k in range(1, K + 1)]
        P += fwd + [flow(np.zeros(2), s, -k * tau) for k in range(1, K + 1)]      # обратная цепочка от цели = ветви кривой переключения
        for k in range(0, K, m):                                                  # веер: другой слой от k-й споры цепочки
            p = np.asarray(x0, float) if k == 0 else fwd[k - 1]
            P += [flow(p, -s, j * tau) for j in range(1, K + 1)]
    P = np.array(P); P = np.unique(np.round(P, 9), axis=0)
    return P[np.all(np.abs(P) < 8.0, axis=1)]


def solve_scattered(P, tau, Rg=0.25, K=60.0, iters=3000, tol=1e-9):
    tri = Delaunay(P); goal = np.hypot(P[:, 0], P[:, 1]) < Rg
    Tg = np.array([T_star(*p) for p in P]); V = np.full(len(P), K); V[goal] = Tg[goal]
    ends = [flow(P, s, tau) for s in (+1.0, -1.0)]
    for it in range(iters):
        lin = LinearNDInterpolator(tri, V); near = NearestNDInterpolator(P, V)
        Vn = V.copy()
        for E in ends:
            q = lin(E); q = np.where(np.isnan(q), near(E) + tau * 0 + K * 0.0 + 1.0, q)   # вне оболочки — ближайшая + штраф 1 (консервативно)
            Vn = np.minimum(Vn, tau + q)
        Vn[goal] = Tg[goal]; ch = float(np.max(V - Vn)); V = Vn
        if ch < tol:
            break
    return tri, V, it + 1


def run_query(P, V, tri, x0, tau, Rg=0.25, T_max=40.0):
    """Жадный агент по интерполированному V; в клетке цели — точная T_star. Возвращает время."""
    lin = LinearNDInterpolator(tri, V); near = NearestNDInterpolator(P, V)
    val = lambda e: (lambda q: float(near(e)) + 1.0 if np.isnan(q) else q)(float(np.ravel(lin(e))[0]))
    p = np.asarray(x0, float); t = 0.0
    while t < T_max:
        if np.hypot(*p) < Rg:
            return t + T_star(*p)
        c = [flow(p, s, tau) for s in (+1.0, -1.0)]; p = min(c, key=lambda e: float(val(e))); t += tau
    return t


def V_query(P, V, tri, x):
    """V в точке x по интерполянту спор."""
    q = float(np.ravel(LinearNDInterpolator(tri, V)(x))[0]); return q if not np.isnan(q) else float(NearestNDInterpolator(P, V)(x)) + 1.0
