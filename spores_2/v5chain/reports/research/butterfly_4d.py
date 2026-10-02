"""research hub-v5chain-research-8: споры-«бабочки» на 4D плоском ДИ (x, y, vx, vy; |ax|, |ay| ≤ 1), цель — коробка ±.1 по всем осям.
Спора c + нормальный отрезок s·n₀, n₀ = (−vy, vx, 0, 0)/|v| (дополнение якобиана [∂Φ/∂t, ∂Φ/∂a] при t→0), |s| ≤ r, m узлов, V_c(s) — линейно.
Из точки y на отрезок споры c с постоянным a — не более одной дуги, всё формулой: w = v_y + v_c, s* = (d×w)/(n₀×w), d = c_pos − y_pos,
t = 2(d + s* n₀)·w/|w|², a = (v_c − v_y)/t; годна при 0 < t ≤ τ, |aᵢ| ≤ 1, |s*| ≤ r. Беллман: V(узел) = min по спорам [t + V_c(s*)] —
пары (узел, спора) и (t, s*) считаются один раз, итерации — векторно. Эталон T* = max(T*_x, T*_y) (оси независимы, коробка управлений)."""
import numpy as np, json, sys, time
from scipy.spatial import cKDTree
sys.path.insert(0, '.'); from v7_faces_di import tstar_box
RHO, BIG, PL, VL = .1, 1e3, 1.5, 1.0
def cr(a, b): return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
class B4:
    def __init__(s, N=20000, r=.15, m=7, tau=.6, seed=0):
        rng = np.random.default_rng(seed); g = np.linspace(-RHO, RHO, 3)
        G = np.array(np.meshgrid(g, g, g, g, indexing='ij')).reshape(4, -1).T
        C = np.r_[G, np.c_[rng.uniform(-PL, PL, (N, 2)), rng.uniform(-VL, VL, (N, 2))]]
        v = C[:, 2:]; nv = np.linalg.norm(v, axis=1); n0 = np.where(nv[:, None] > 1e-3, np.c_[-v[:, 1], v[:, 0]] / np.maximum(nv, 1e-9)[:, None], np.array([1., 0]))
        s.C, s.n0, s.K, s.r, s.m, s.tau = C, n0, len(C), r, m, tau; s.sn = np.linspace(-r, r, m)
        s.nodes = (C[:, None, :] + np.concatenate([n0[:, None, :] * s.sn[None, :, None], np.zeros((s.K, m, 2))], 2)).reshape(-1, 4)
        s.ingoal = (np.abs(s.nodes) <= RHO + 1e-12).all(1); s.V = np.where(s.ingoal, 0., BIG)
        s.scale = np.array([1 / (tau * VL + tau * tau / 2 + r), 1 / (tau * VL + tau * tau / 2 + r), 1 / tau, 1 / tau])   # соседи: |Δpos|, |Δv| ≤ 1 в масштабе
        s.tree = cKDTree(C * s.scale)
    def arcs(s, Y, k):
        """дуги из точек Y (q,4) на отрезки спор k (q,) → t, s*, a (inf, если нет)."""
        c = s.C[k]; n0 = s.n0[k]; w = Y[:, 2:] + c[:, 2:]; d = c[:, :2] - Y[:, :2]
        with np.errstate(divide='ignore', invalid='ignore'):
            ss = -cr(d, w) / cr(n0, w); t = 2 * np.einsum('qi,qi->q', d + ss[:, None] * n0, w) / np.einsum('qi,qi->q', w, w); a = (c[:, 2:] - Y[:, 2:]) / t[:, None]
        ok = np.isfinite(ss) & (np.abs(ss) <= s.r) & (t > 1e-6) & (t <= s.tau) & (np.abs(a) <= 1 + 1e-12).all(1)
        return np.where(ok, t, np.inf), ss, a
    def pairs(s, Y, own=None, chunk=1500):
        if len(Y) > chunk:
            R = [s.pairs(Y[i:i + chunk], None if own is None else own[i:i + chunk]) for i in range(0, len(Y), chunk)]
            off = np.arange(0, len(Y), chunk); return tuple(np.concatenate([x[j] + (off[i] if j == 0 else 0) for i, x in enumerate(R)]) for j in range(5))
        L = s.tree.query_ball_point(Y * s.scale, np.sqrt(2) + 1e-9)
        qi = np.repeat(np.arange(len(Y)), [len(l) for l in L]); k = np.concatenate([np.array(l, int) for l in L]) if len(qi) else np.zeros(0, int)
        if own is not None: keep = k != own[qi]; qi, k = qi[keep], k[keep]
        t, ss, a = s.arcs(Y[qi], k); f = np.isfinite(t); return qi[f], k[f], t[f], ss[f], a[f]
    def Vc(s, k, ss):
        Vn = s.V.reshape(s.K, s.m); fpos = (ss + s.r) / (2 * s.r) * (s.m - 1); j = np.clip(np.floor(fpos).astype(int), 0, s.m - 2); b = fpos - j
        return (1 - b) * Vn[k, j] + b * Vn[k, j + 1]
    def solve(s, it=100):
        own = np.repeat(np.arange(s.K), s.m); t0 = time.time()
        s.P = s.pairs(s.nodes, own); print('pairs', len(s.P[0]), 'per node', round(len(s.P[0]) / len(s.nodes), 1), 'sec', round(time.time() - t0), flush=True)
        qi, k, t, ss, _ = s.P
        for n in range(it):
            J = t + s.Vc(k, ss); Vn = np.full(len(s.nodes), BIG); np.minimum.at(Vn, qi, J); Vn = np.where(s.ingoal, 0., np.minimum(Vn, BIG))
            d = np.max(np.abs(Vn - s.V)); s.V = Vn
            if d < 1e-7: break
        s.it = n; return s
    def best(s, Y):
        qi, k, t, ss, a = s.pairs(Y); J = t + s.Vc(k, ss); V = np.full(len(Y), np.inf); A = np.zeros((len(Y), 2)); T = np.zeros(len(Y))
        o = np.lexsort((J, qi)); first = np.r_[True, qi[o][1:] != qi[o][:-1]]; sel = o[first]
        V[qi[sel]] = J[sel]; A[qi[sel]] = a[sel]; T[qi[sel]] = t[sel]
        inside = (np.abs(Y) <= RHO + 1e-9).all(1); V[inside] = 0; return V, A, T
    def rollout(s, Q, dt=.06, tmax=25.):
        Y = Q.copy(); n = len(Y); T = np.full(n, np.inf); act = np.ones(n, bool); t = 0.; pa = np.full((n, 2), np.nan); sw = np.zeros(n)
        while t < tmax and act.any():
            ia = np.nonzero(act)[0]; V, A, _ = s.best(Y[ia]); dead = ~np.isfinite(V) | (V >= BIG / 2); act[ia[dead]] = False
            ia, A = ia[~dead], A[~dead]
            sw[ia] += np.isfinite(pa[ia, 0]) & (np.abs(A - pa[ia]).max(1) > .5); pa[ia] = A
            for _ in range(6):
                h = dt / 6; Y[ia, :2] += Y[ia, 2:] * h + A * h * h / 2; Y[ia, 2:] += A * h
                g = (np.abs(Y[ia]) <= RHO + 1e-9).all(1); T[ia[g]] = t + h * (_ + 1); act[ia[g]] = False; keep = ~g; ia, A = ia[keep], A[keep]
            t += dt
        return T, sw
if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = time.time(); S = B4(N=N).solve(); print('spores', S.K, 'nodes', len(S.nodes), 'iters', S.it, 'BIG', round(float((S.V >= BIG / 2).mean()), 3), 'sec', round(time.time() - t0), flush=True)
    rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-1, 1, (300, 2)), rng.uniform(-.7, .7, (300, 2))]
    Ts = np.maximum(tstar_box(Q[:, 0], Q[:, 2], RHO), tstar_box(Q[:, 1], Q[:, 3], RHO)); ok = Ts > .05; Q, Ts = Q[ok], Ts[ok]
    V, _, _ = S.best(Q); m = np.isfinite(V) & (V < BIG / 2); r = V[m] / Ts[m]
    print(json.dumps(dict(N=N, V_cover=round(float(m.mean()), 3), V_over_Tstar=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 3), max=round(float(r.max()), 3))), ensure_ascii=False), flush=True)
    T, sw = S.rollout(Q[:150]); f = np.isfinite(T); rr = T[f] / Ts[:150][f]
    print(json.dumps(dict(N=N, agent='dt', reach=round(float(f.mean()), 3), T_over_Tstar=dict(mean=round(float(rr.mean()), 4), med=round(float(np.median(rr)), 4), max=round(float(rr.max()), 3)) if f.any() else None,
                          sw_med=float(np.median(sw)), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
