"""V по клеткам (source_doc §4): V хранится в центрах спор решётки шага h. Q_k(c) = τ + V(выход клетки слоя k), выход = поток центра на τ;
V на выходе — билинейно по соседним спорам (выход лежит внутри соседей). V(c) = min_k Q_k(c), цель — клетка радиуса R_goal, в ней V = T* (точная формула). Итерации Беллмана до сходимости."""
import numpy as np
from .cell import flow
from .agent import T_star


def solve_V(h, lim=4.0, tau=None, R_goal=None, umax=1.0, iters=2000, tol=1e-9):
    tau = h / 2 if tau is None else tau
    R_goal = 2 * h if R_goal is None else R_goal
    xs = np.arange(-lim, lim + 1e-9, h); n = len(xs)
    X, Vv = np.meshgrid(xs, xs, indexing='ij')
    ends = [flow(np.stack([X, Vv], -1), s * umax, tau) for s in (+1, -1)]
    goal = np.hypot(X, Vv) < R_goal
    Tg = np.vectorize(T_star)(X, Vv)                                       # клетка цели — своя точная формула (source_doc §4)
    K = 3 * float(Tg.max()); V = np.full(X.shape, K); V[goal] = Tg[goal]                # верхняя оценка K, итерации идут вниз (как в dd_atlas)

    def interp(V, P):
        fx = (P[..., 0] + lim) / h; fv = (P[..., 1] + lim) / h
        inside = (fx >= 0) & (fv >= 0) & (fx <= n - 1) & (fv <= n - 1)
        fx = np.clip(fx, 0, n - 1 - 1e-9); fv = np.clip(fv, 0, n - 1 - 1e-9)
        i, j = fx.astype(int), fv.astype(int); u, w = fx - i, fv - j
        W = [(1 - u) * (1 - w), u * (1 - w), (1 - u) * w, u * w]; Q = [V[i, j], V[i + 1, j], V[i, j + 1], V[i + 1, j + 1]]
        r = sum(wk * qk for wk, qk in zip(W, Q))                                       # билинейно, все узлы конечны (старт с K)
        r = np.where(inside, r, K)
        return r

    for it in range(iters):
        Vn = np.minimum(V, np.minimum(*[tau + interp(V, e) for e in ends])); Vn[goal] = Tg[goal]
        ch = float(np.max(V - Vn)); V = Vn
        if ch < tol:
            break
    return xs, V, it + 1


def V_interp(xs, V, x, v):
    h = xs[1] - xs[0]; n = len(xs)
    fx = np.clip((x - xs[0]) / h, 0, n - 1 - 1e-9); fv = np.clip((v - xs[0]) / h, 0, n - 1 - 1e-9)
    i, j = int(fx), int(fv); u, w = fx - i, fv - j
    return (1 - u) * (1 - w) * V[i, j] + u * (1 - w) * V[i + 1, j] + (1 - u) * w * V[i, j + 1] + u * w * V[i + 1, j + 1]
