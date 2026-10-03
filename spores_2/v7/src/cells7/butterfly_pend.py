"""Споры-«бабочки» на маятнике φ̈ = sin φ + u, |u| ≤ UM (цель — верх, φ отсчитан от верха; hub-v5chain-research-8, очередь №2 п.2).
Спора c: отрезок c + s·e⊥, e⊥ ⟂ дрейфу f(c,0) = (ω, sin φ), |s| ≤ r, m узлов V. Дуга из y в точку p отрезка с постоянным u:
H_u = ω²/2 + cos φ − uφ сохраняется ⇒ u = [(ω_p²−ω_y²)/2 + cos φ_p − cos φ_y] / Δφ ЗАМКНУТО (Δφ — свёрнутая разность), t — rk4-прогон по u + Ньютон
по точке ближайшего подхода. Беллман — как в `butterfly_di.ButterflyFast` (пары строятся один раз)."""
import numpy as np
from scipy.spatial import cKDTree
BIG, UM, PER = 1e3, 0.3, 2 * np.pi
RX, RW = .02 * np.pi, .02 * 4.0


def wrap(x): return (x + np.pi) % PER - np.pi
def f(z, u): return np.stack([z[:, 1], np.sin(z[:, 0]) + u], 1)
def rk4(z, u, h):
    k1 = f(z, u); k2 = f(z + h[:, None] / 2 * k1, u); k3 = f(z + h[:, None] / 2 * k2, u); k4 = f(z + h[:, None] * k3, u)
    return z + h[:, None] / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def ingoal(z): return (np.abs(wrap(z[..., 0])) <= RX + 1e-9) & (np.abs(z[..., 1]) <= RW + 1e-9)


def arc_times(Y, P, tau, ng=24, newton=3):
    ng = max(ng, int(np.ceil(1.5 * tau / .05)))                          # шаг rk4 прогона ≤ .05 с
    """Y,P (N,2): физ. координаты; (t, u, ok): дуга с постоянным u из Y в P за t ≤ tau."""
    N = len(Y); dphi = wrap(P[:, 0] - Y[:, 0]); ok = np.abs(dphi) > 1e-6
    u = np.where(ok, ((P[:, 1] ** 2 - Y[:, 1] ** 2) / 2 + np.cos(P[:, 0]) - np.cos(Y[:, 0])) / np.where(ok, dphi, 1), 0.)
    den = Y[:, 1] + P[:, 1]; t0 = np.where(np.abs(den) > 1e-9, 2 * dphi / np.where(np.abs(den) > 1e-9, den, 1), np.inf)       # трапеция — грубая оценка
    ok &= (np.abs(u) <= UM) & (t0 > 0) & (t0 <= 1.5 * tau)
    t = np.full(N, np.inf); idx = np.nonzero(ok)[0]
    if not len(idx): return t, u, ok & False
    y, p, uu = Y[idx], P[idx], u[idx]; tmax = 1.5 * tau; h = np.full(len(idx), tmax / ng); z = y.copy(); best = np.full(len(idx), np.inf); tb = np.zeros(len(idx)); zb = y.copy()
    for i in range(1, ng + 1):
        z = rk4(z, uu, h); d = np.hypot(wrap(z[:, 0] - p[:, 0]), z[:, 1] - p[:, 1]); b = d < best; best = np.where(b, d, best); tb = np.where(b, i * h, tb); zb = np.where(b[:, None], z, zb)
    for _ in range(newton):                                             # проекция на касательную в точке ближайшего подхода
        v = f(zb, uu); dz = np.stack([wrap(p[:, 0] - zb[:, 0]), p[:, 1] - zb[:, 1]], 1); dt = np.sum(dz * v, 1) / np.maximum(np.sum(v * v, 1), 1e-12)
        tb = tb + dt; zb = rk4(zb, uu, dt)
    d = np.hypot(wrap(zb[:, 0] - p[:, 0]), zb[:, 1] - p[:, 1]); good = (d < 2e-4) & (tb > 1e-6) & (tb <= tau)
    t[idx[good]] = tb[good]; ok2 = np.zeros(N, bool); ok2[idx[good]] = True
    return t, u, ok2


class ButterflyPend:
    def __init__(s, N=3000, r=.1, m=7, tau=.5, ns=21, seed=0, goal_spores=3, win=None, extra=None):
        rng = np.random.default_rng(seed)
        g = np.linspace(-1, 1, goal_spores); gc = np.array([(a * RX, b * RW) for a in g for b in g])
        s.C = np.r_[gc, np.stack([rng.uniform(-np.pi, np.pi, N), rng.uniform(-4, 4, N)], 1)]
        if extra is not None: s.C = np.r_[s.C, extra]                     # extra — дополнительные споры (заполнение дыр покрытия)
        s.K = len(s.C); s.ngoal = len(gc)
        s.win = win; s.r, s.m, s.tau = r, m, tau; s.sn = np.linspace(-r, r, m); s.sq = np.linspace(-r, r, ns)
        fl = np.stack([s.C[:, 1], np.sin(s.C[:, 0])], 1); nr = np.maximum(np.hypot(fl[:, 0], fl[:, 1]), 1e-9); s.E = np.stack([-fl[:, 1], fl[:, 0]], 1) / nr[:, None]    # e⊥ ⟂ дрейфу
        s.V = np.full((s.K, m), BIG); s.nodeP = s.C[:, None, :] + s.sn[None, :, None] * s.E[:, None, :]          # узлы отрезков (K,m,2)
        s.ing = ingoal(s.nodeP); s.V[s.ing] = 0.
        sh = np.array([-PER, 0., PER]); s.tree = cKDTree(np.concatenate([s.C + np.array([a, 0]) for a in sh])); 
    def cand(s, y):
        rad = (s.win + s.r + .05) if s.win else s.tau * (abs(y[1]) + 1.5) + s.r + .05; k = np.unique(np.array(s.tree.query_ball_point(y, rad)) % s.K)
        return k
    def pairs_for(s, y, exclude=-1):
        k = s.cand(y); k = k[k != exclude]
        if not len(k): return None
        P = s.C[k][:, None, :] + s.sq[None, :, None] * s.E[k][:, None, :]; kk = np.repeat(k, len(s.sq)); qq = np.tile(np.arange(len(s.sq)), len(k))
        t, u, ok = arc_times(np.broadcast_to(y, (len(kk), 2)).copy(), P.reshape(-1, 2), s.tau)
        if s.win: dist = np.hypot(wrap(P.reshape(-1, 2)[:, 0] - y[0]), P.reshape(-1, 2)[:, 1] - y[1]); ok &= dist <= s.win
        return kk[ok], qq[ok], t[ok], u[ok]
    def build_pairs(s, log=None):
        nodes, ks, js, ws, ts = [], [], [], [], []
        pos = (s.sq + s.r) / (2 * s.r) * (s.m - 1); j0q = np.clip(np.floor(pos).astype(int), 0, s.m - 2); wq = pos - j0q
        for i in range(s.K):
            for j in range(s.m):
                if s.ing[i, j]: continue
                pr = s.pairs_for(s.nodeP[i, j], exclude=i)
                if pr is None or not len(pr[0]): continue
                k, q, t, u = pr; nodes.append(np.full(len(k), i * s.m + j)); ks.append(k); js.append(j0q[q]); ws.append(wq[q]); ts.append(t)
            if log and i % 200 == 0: log('pairs spore', i, sum(len(a) for a in nodes))
        s.pn, s.pk, s.pj, s.pw, s.pt = (np.concatenate(x) for x in (nodes, ks, js, ws, ts))
        s.pidx = s.pk * s.m + s.pj; s.starts = np.r_[0, np.nonzero(np.diff(s.pn))[0] + 1]; s.pnode = s.pn[s.starts]
    def solve(s, it=5000, tol=1e-9, log=None):
        s.build_pairs(log); V = s.V.ravel().copy(); G = s.ing.ravel()
        for n in range(it):
            val = s.pt + (1 - s.pw) * V[s.pidx] + s.pw * V[s.pidx + 1]; mn = np.minimum.reduceat(val, s.starts)
            Vn = V.copy(); Vn[s.pnode] = np.minimum(V[s.pnode], mn); Vn[G] = 0.; d = np.max(np.abs(np.minimum(Vn, BIG) - np.minimum(V, BIG))); V = Vn
            if log and n % 100 == 0: log('it', n, d)
            if d < tol: break
        s.V = V.reshape(s.K, s.m); s.n_it = n; s.npairs = len(s.pt); return s
    def Vc(s, k, q):
        pos = (s.sq[q] + s.r) / (2 * s.r) * (s.m - 1); j0 = np.clip(np.floor(pos).astype(int), 0, s.m - 2); w = pos - j0; return (1 - w) * s.V[k, j0] + w * s.V[k, j0 + 1]
    def best(s, y):
        y = np.array(y, float); y[0] = wrap(y[0])
        if ingoal(y)[()]: return 0., None
        pr = s.pairs_for(y)
        if pr is None or not len(pr[0]): return BIG, None
        k, q, t, u = pr; J = t + s.Vc(k, q); i = int(np.argmin(J)); return J[i], (k[i], q[i], t[i], u[i])
    def Vq(s, Y): return np.array([s.best(y)[0] for y in Y])
    def rollout(s, y0, dt=.06, tmax=40., sub=4, eps=0.):
        y = np.array(y0, float); t = 0.; pu = None; sw = 0
        while t < tmax:
            if ingoal(y): return t, sw
            J, b = s.best(y)
            if b is None or J >= BIG / 2: return np.inf, sw
            _, _, ta, u = b; h = min(dt, ta)
            if pu is not None and abs(u - pu) > .5 * UM: sw += 1
            pu = u; z = y[None, :]
            for _ in range(sub):
                z = rk4(z, np.array([u]), np.array([h / sub])); t += h / sub
                if ingoal(z[0]): return t, sw
            y = z[0].copy(); y[0] = wrap(y[0])                                   # агент накручивает обороты — φ в [−π,π)
        return np.inf, sw
