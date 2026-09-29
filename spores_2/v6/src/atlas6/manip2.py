"""Манипулятор 2 звена, кинематика q̇ = u, |u_i| ≤ 1, состояние — тор T² (ступень 1, research/manipulator_plan.md). 8 слоёв: 4 угла + 4 середины рёбер.
V в узлах тора (шаг h), выход клетки — билинейно (периодично); цель — |Δq_i| < Rg (по L∞). Точный эталон: T* = max_i |Δq_i mod 2π| (суставы независимы)."""
import numpy as np

LAYERS2 = [(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0) if (a, b) != (0.0, 0.0)]
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def T_star2(q):
    q = np.asarray(q, float); return np.max(np.abs(wrap(q)), axis=-1)


def _seg_disc(p, q, c, r):
    d = q - p; t = np.clip(np.dot(c - p, d) / max(np.dot(d, d), 1e-12), 0, 1); return np.linalg.norm(p + t * d - c) <= r


def collide2(q, obstacles, L1=1.0, L2=1.0):
    """Звенья-отрезки (L1, L2 из начала) против дисков рабочего пространства obstacles = [((cx, cy), r)]."""
    e = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])]); t = e + L2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
    return any(_seg_disc(np.zeros(2), e, np.array(c), r) or _seg_disc(e, t, np.array(c), r) for c, r in obstacles)


def solve_manip2(h=2 * np.pi / 48, tau=None, Rg=None, iters=5000, tol=1e-10, obstacles=(), K=30.0):
    n = int(round(2 * np.pi / h)); h = 2 * np.pi / n; tau = h if tau is None else tau; Rg = h * 0.5 if Rg is None else Rg
    g = -np.pi + h * np.arange(n); Q1, Q2 = np.meshgrid(g, g, indexing='ij')
    goal = (np.abs(wrap(Q1)) < Rg) & (np.abs(wrap(Q2)) < Rg)
    obstacles = list(obstacles); K = K if obstacles else 4 * np.pi
    occ = np.array([[collide2((a, b), obstacles) for b in g] for a in g]) if obstacles else np.zeros(Q1.shape, bool)
    V = np.full(Q1.shape, K); V[goal] = 0.0; V[occ] = K
    bad = []                                                                     # ребро запрещено, если середина или конец в столкновении
    for u1, u2 in LAYERS2:
        b = np.zeros(Q1.shape, bool)
        if obstacles:
            for f in (0.5, 1.0):
                b |= np.array([[collide2((a + u1 * tau * f, c + u2 * tau * f), obstacles) for c in g] for a in g])
        bad.append(b)

    def interp(V, a, b):
        f1 = ((a + np.pi) / h) % n; f2 = ((b + np.pi) / h) % n; i, j = np.floor(f1).astype(int), np.floor(f2).astype(int); u, w = f1 - i, f2 - j
        i2, j2 = (i + 1) % n, (j + 1) % n
        return (1 - u) * (1 - w) * V[i, j] + u * (1 - w) * V[i2, j] + (1 - u) * w * V[i, j2] + u * w * V[i2, j2]
    for it in range(iters):
        Vn = V.copy()
        for (u1, u2), b in zip(LAYERS2, bad):
            Vn = np.minimum(Vn, np.where(b, K, tau + interp(V, Q1 + u1 * tau, Q2 + u2 * tau)))
        Vn[goal] = 0.0; Vn[occ] = K; ch = float(np.max(V - Vn)); V = Vn
        if ch < tol:
            break
    return dict(g=g, V=V, h=h, n=n, tau=tau, Rg=Rg, iters=it + 1, occ=occ)


def V_at2(A, q):
    h, n, V = A['h'], A['n'], A['V']; f = ((np.asarray(q, float) + np.pi) / h) % n; i, j = int(f[0]), int(f[1]); u, w = f[0] - i, f[1] - j
    return (1 - u) * (1 - w) * V[i, j] + u * (1 - w) * V[(i + 1) % n, j] + (1 - u) * w * V[i, (j + 1) % n] + u * w * V[(i + 1) % n, (j + 1) % n]
