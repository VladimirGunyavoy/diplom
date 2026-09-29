"""Атлас v6 для дифдрайва (x, y, θ), U = ромб |v|+|ω| ≤ 1: 4 слоя (прямая ±, поворот на месте ±). V в спорах сетки (h по x,y; hθ по θ, θ периодично).
Q_k(c) = τ + V(flow_k(c, τ)), V в выходе — трилинейно по 8 спорам. Цель — клетка |xy| < R_goal, |θ| < Rθ: V = 0 (точная формула не известна — эталон отдельно)."""
import numpy as np
from .dd3 import flow

LAYERS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]                      # ромб
RECT = LAYERS + [(sv, sw) for sv in (1.0, -1.0) for sw in (1.0, -1.0)]         # прямоугольник |v|,|ω| ≤ 1: + 4 дуги-вершины


def blocked(E, obst, rr=0.0):
    E = np.asarray(E, float); m = np.zeros(E.shape[:-1], bool)
    for cx, cy, r in obst:
        m |= np.hypot(E[..., 0] - cx, E[..., 1] - cy) < r + rr
    return m


def solve_dd(h=0.25, lim=3.0, nth=None, tau=None, R_goal=None, Rth=None, iters=3000, tol=1e-9, layers=None, obstacles=(), robot_r=0.0, K=30.0):
    """obstacles — диски (cx, cy, r) в xy (вытянуты по всем θ); ребро запрещено, если поток на τ/2, τ или узел ближе r+robot_r. С препятствиями старт итерации K (верхняя оценка 2π+r неверна)."""
    layers = LAYERS if layers is None else layers
    obst = [tuple(map(float, o)) for o in obstacles]
    nth = nth or 4 * int(round(np.pi / (2 * h))); hth = 2 * np.pi / nth
    tau = h if tau is None else tau                                        # τ=h лучше h/2 (research 2026-09-30: V/эталон 1.037 vs 1.052); √h — хуже
    R_goal = h if R_goal is None else R_goal; Rth = hth if Rth is None else Rth
    xs = np.arange(-lim, lim + 1e-9, h); n = len(xs); ths = -np.pi + hth * np.arange(nth)
    X, Y, TH = np.meshgrid(xs, xs, ths, indexing='ij')
    P = np.stack([X, Y, TH], -1)
    ends = [flow(P, vw, tau) for vw in layers]
    dth = (TH + np.pi) % (2 * np.pi) - np.pi
    goal = (np.hypot(X, Y) < R_goal) & (np.abs(dth) < Rth)
    hit = lambda E: blocked(E, obst, robot_r)
    ub = (lambda E: K + 0 * E[..., 0]) if obst else (lambda E: 2 * np.pi + np.hypot(E[..., 0], E[..., 1]))                     # верхняя оценка: повернуть, проехать, повернуть
    V = ub(P); V[goal] = 0.0; V[hit(P)] = K
    bad = [hit(P) | hit(flow(P, vw, tau / 2)) | hit(flow(P, vw, tau)) for vw in layers]                                                     # итерация Беллмана идёт вниз от верхней оценки — без бесконечностей

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
        for E, b in zip(ends, bad):
            Vn = np.minimum(Vn, np.where(b, K, tau + interp(V, E)))
        Vn[goal] = 0.0; Vn[hit(P)] = K; ch = float(np.max(V - Vn)); V = Vn
        if ch < tol:
            break
    return dict(xs=xs, ths=ths, V=V, iters=it + 1, h=h, hth=hth, tau=tau, layers=layers, obst=obst, robot_r=robot_r, lim=lim, R_goal=R_goal, Rth=Rth)


def V_at(A, p):
    """V в произвольной позе — трилинейно по 8 спорам (θ периодично)."""
    xs, V, h, hth, lim = A['xs'], A['V'], A['h'], A['hth'], A['lim']; n, nth = len(xs), V.shape[2]
    fx = np.clip((p[0] + lim) / h, 0, n - 1 - 1e-9); fy = np.clip((p[1] + lim) / h, 0, n - 1 - 1e-9); ft = ((p[2] + np.pi) / hth) % nth
    i, j, k = int(fx), int(fy), int(ft); u, w, s = fx - i, fy - j, ft - k; k2 = (k + 1) % nth
    return sum(a * b * c * V[i + di, j + dj, kk] for di, a in ((0, 1 - u), (1, u)) for dj, b in ((0, 1 - w), (1, w)) for kk, c in ((k, 1 - s), (k2, s)))


def rollout(A, p0, dt=None, T_max=30.0):
    """Агент: на каждом шаге слой с min V(flow(p, dt)) (ромб: 4 вершины). Возвращает (путь, время, дошёл)."""
    dt = A['tau'] if dt is None else dt; p = np.asarray(p0, float); path = [p]; t = 0.0
    while t < T_max:
        d = (p[2] + np.pi) % (2 * np.pi) - np.pi
        if np.hypot(p[0], p[1]) < A['R_goal'] and abs(d) < A['Rth']:
            return np.array(path), t, True
        cands = [flow(p, vw, dt) for vw in A['layers'] if not (blocked(flow(p, vw, dt / 2), A['obst'], A['robot_r']) or blocked(flow(p, vw, dt), A['obst'], A['robot_r']))]
        if not cands:
            return np.array(path), t, False
        p = min(cands, key=lambda e: V_at(A, e)); path.append(p); t += dt
    return np.array(path), t, False


def corridor(A, p0, dt=None, min_len=0.15):
    """Если невязка ≥ 1e-6 (сегментов < 3 не хватает на 3 степени свободы), min_len уменьшается (0.15 → 0.05 → 0)."""
    for dd in ((A['tau'] / 2, A['tau'] / 4) if dt is None else (dt,)):                  # шаг агента мельче, если SLSQP не сошёлся
        for m, pad in ((min_len, False), (0.05, False), (0.0, False), (min_len, True)):
            res = _corridor(A, p0, dd, m, pad)
            if res[2] < 1e-6:
                return res
    return res


def _corridor(A, p0, dt, min_len, pad=False):
    """Коридор (dt_i, слой_i): слои жадного агента, RLE (короткие сливаются), длительности — SLSQP: min Σdt при конец = точка цели (θ без свёртки)."""
    from scipy.optimize import minimize
    dt = A['tau'] / 2 if dt is None else dt; p = np.asarray(p0, float); seq = []; t = 0.0
    while t < 30.0:
        d = (p[2] + np.pi) % (2 * np.pi) - np.pi
        if np.hypot(p[0], p[1]) < A['R_goal'] and abs(d) < A['Rth']:
            break
        k = int(np.argmin([V_at(A, flow(p, vw, dt)) for vw in A['layers']])); p = flow(p, A['layers'][k], dt); t += dt
        if seq and seq[-1][1] == k:
            seq[-1][0] += dt
        else:
            seq.append([dt, k])
    out = []
    for d, k in seq:
        if out and (d < min_len or out[-1][1] == k):
            out[-1][0] += d
        else:
            out.append([d, k])
    if not out:                                                                  # старт уже в клетке цели
        return [], 0.0, 0.0, 0.0
    if pad:                                                                      # добавка коротких сегментов всех слоёв: степеней свободы ≥ 3
        out = out + [[0.02, k] for k in range(len(A['layers']))]
    ks = [k for _, k in out]; d0 = np.array([d for d, _ in out]); target = np.array([0.0, 0.0, 2 * np.pi * round(p[2] / (2 * np.pi))])

    def ep(d):
        q = np.asarray(p0, float)
        for di, k in zip(d, ks):
            q = flow(q, A['layers'][k], max(di, 0.0))
        return q - target
    def clear(d):                                                                # зазор до дисков в 8 точках каждого сегмента (≥ 0 — свободно)
        q = np.asarray(p0, float); g = []
        for di, k in zip(d, ks):
            di = max(di, 0.0)
            for f in np.linspace(0.125, 1.0, 8):
                e = flow(q, A['layers'][k], f * di)
                g += [np.hypot(e[0] - cx, e[1] - cy) - r - A['robot_r'] for cx, cy, r in A['obst']]
            q = flow(q, A['layers'][k], di)
        return np.array(g)
    cons = [{'type': 'eq', 'fun': ep}] + ([{'type': 'ineq', 'fun': clear}] if A['obst'] else [])
    r = minimize(lambda d: d.sum(), d0, jac=lambda d: np.ones_like(d), bounds=[(0, None)] * len(d0),
                 constraints=cons, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-12})
    return [(float(d), k) for d, k in zip(r.x, ks)], float(r.x.sum()), float(np.linalg.norm(ep(r.x))), float(d0.sum())


def corridor_collides(A, p0, cor, ds=0.02):
    """Число точек пути коридора [(dt, k)] внутри препятствий (с запасом robot_r), шаг по времени ds."""
    q = np.asarray(p0, float); n = 0
    for d, k in cor:
        for t in np.arange(ds, d + ds, ds):
            n += int(blocked(flow(q, A['layers'][k], min(t, d)), A['obst'], A['robot_r']))
        q = flow(q, A['layers'][k], d)
    return n
