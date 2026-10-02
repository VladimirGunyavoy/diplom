"""research hub-v5chain-research-9: споры-«бабочки» на маятнике (продолжение butterfly_di.py; идея пользователя 2026-10-02).
Маятник φ̈ = sin φ + u, |u| ≤ UM = .3, цель |φ|,|ω| ≤ .1 (верх), чистое время. Спора = точка c + отрезок c + s·n, n ⟂ дрейфу f(c,0) = (ω, sin φ),
|s| ≤ r, m узлов V на отрезке. Обратная задача (из y в точку p отрезка одной дугой с постоянным u): неизвестные (t, u), 2 уравнения —
Ньютон по ТОЧНОМУ потоку (rk4), старт с формулы ДИ (t0 = 2Δφ/(ω_y+ω_p), u0 = Δω/t0 − sin φ_сер); ∂Φ/∂t = f — даром, ∂Φ/∂u — разностью.
Беллман внутри карты: V(y) = min по спорам и s [t + V_c(s)], V_c линейно по узлам; пары (узел → спора, s, t) считаются один раз, дальше Якоби.
Эталон — сетка HJB (pend_cost_grid.Grid, bang ±1): V_сетки и её агент."""
import numpy as np, json, sys, time, os
sys.path.insert(0, '.')
from pend_cost_grid import Grid, step, wrap, UM
RHO, BIG = .1, 1e3
WS = float(os.environ.get('WS', 3.5)); TL = float(os.environ.get('TL', 0))                          # TL > 0: предел времени дуги отдельно от τ (τ — только размер окна поиска)                                          # споры: |ω| ≤ WS

def flow(x, w, u, t, n=4 if TL <= .5 else 8):
    h = t / n
    for _ in range(n): x, w = step(x, w, u, h)
    return x, w

class ButterflyPend:
    def __init__(s, N=2000, r=.1, m=7, tau=.5, ns=19, seed=0):
        rng = np.random.default_rng(seed)
        g = np.linspace(-RHO, RHO, 3); gc = np.array([(a, b) for a in g for b in g])
        s.C = np.r_[gc, np.stack([rng.uniform(-np.pi, np.pi, N), rng.uniform(-WS, WS, N)], 1)]; s.K = len(s.C)
        n = np.stack([-np.sin(s.C[:, 0]), s.C[:, 1]], 1); nn = np.linalg.norm(n, axis=1)
        s.n = np.where(nn[:, None] > .05, n / np.maximum(nn, 1e-12)[:, None], np.array([0., 1.]))      # у равновесий — вдоль ω
        s.r, s.m, s.tau = r, m, tau; s.tl = TL or tau; s.sn = np.linspace(-r, r, m); s.sq = np.linspace(-r, r, ns)
        s.P = s.C[:, None, :] + s.sn[None, :, None] * s.n[:, None, :]                                    # узлы (K, m, 2)
        s.ingoal = (np.abs(wrap(s.P[..., 0])) <= RHO) & (np.abs(s.P[..., 1]) <= RHO)
        s.V = np.full((s.K, m), BIG); s.V[s.ingoal] = 0.
        j0 = np.clip(np.searchsorted(s.sn, s.sq, side='right') - 1, 0, m - 2); s.qj0 = j0; s.qa = (s.sq - s.sn[j0]) / (s.sn[j0 + 1] - s.sn[j0])
    def pairs(s, Y, own=None, chunk=400):
        """дуги из точек Y (n,2) в точки отрезков всех спор: (iy, k, q, t, u) — только годные (0 < t ≤ τ, |u| ≤ UM, невязка < 1e-7)."""
        out = [[], [], [], [], []]; tau = s.tau
        for a in range(0, len(Y), chunk):
            y = Y[a:a + chunk]; dx = wrap(s.C[None, :, 0] - y[:, None, 0]); dw = s.C[None, :, 1] - y[:, None, 1]
            box = (np.abs(dx) <= (np.abs(y[:, None, 1]) + .7) * tau + s.r) & (np.abs(dw) <= 1.3 * tau + s.r)
            if own is not None: box[np.arange(len(y)), own[a:a + chunk]] = False
            iy, k = np.nonzero(box)
            if not len(iy): continue
            iy = np.repeat(iy, len(s.sq)); k = np.repeat(k, len(s.sq)); q = np.tile(np.arange(len(s.sq)), len(iy) // len(s.sq))
            x, w = y[iy, 0], y[iy, 1]; p = s.C[k] + s.sq[q, None] * s.n[k]; xp = x + wrap(p[:, 0] - x); wp = p[:, 1]
            den = w + wp; g = np.abs(den) > 1e-6; t = np.where(g, 2 * (xp - x) / np.where(g, den, 1.), -1.)
            g &= (t > 1e-4) & (t <= 1.3 * s.tl); u = np.where(g, (wp - w) / np.where(g, t, 1.) - np.sin(x + (xp - x) / 2), 9.); g &= np.abs(u) <= UM + .25
            iy, k, q, x, w, xp, wp, t, u = (v[g] for v in (iy, k, q, x, w, xp, wp, t, u))
            for _ in range(6):
                X, Wv = flow(x, w, u, t); rx, rw = X - xp, Wv - wp; fx, fw = Wv, np.sin(X) + u
                Xu, Wu = flow(x, w, u + 1e-5, t); gx, gw = (Xu - X) / 1e-5, (Wu - Wv) / 1e-5
                det = fx * gw - fw * gx; det = np.where(np.abs(det) > 1e-12, det, 1e-12)
                t = np.clip(t - (rx * gw - rw * gx) / det, 1e-5, 2 * s.tl); u = np.clip(u - (fx * rw - fw * rx) / det, -2., 2.)
            X, Wv = flow(x, w, u, t); ok = (np.hypot(X - xp, Wv - wp) < 1e-7) & (t > 1e-4) & (t <= s.tl) & (np.abs(u) <= UM + 1e-9)
            for o, v in zip(out, (iy[ok] + a, k[ok], q[ok], t[ok], u[ok])): o.append(v)
        return [np.concatenate(o) if o else np.zeros(0) for o in out]
    def build(s):
        Y = s.P.reshape(-1, 2); own = np.repeat(np.arange(s.K), s.m)
        iy, k, q, t, u = s.pairs(Y, own); o = np.argsort(iy, kind='stable'); s.e = [v[o] for v in (iy.astype(int), k.astype(int), q.astype(int), t)]
        return s
    def solve(s, it=3000, tol=1e-7):
        iy, k, q, t = s.e; j0, a = s.qj0[q], s.qa[q]; V = s.V.reshape(-1).copy(); fixed = s.ingoal.reshape(-1)
        st = np.flatnonzero(np.r_[True, iy[1:] != iy[:-1]]); nodes = iy[st]
        for n in range(it):
            V0, V1 = V[k * s.m + j0], V[k * s.m + j0 + 1]
            val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - 1e-9)) | ((V1 >= BIG / 2) & (a > 1e-9))] = BIG
            new = V.copy(); new[nodes] = np.minimum(V[nodes], np.minimum.reduceat(val, st)); new[fixed] = 0.
            d = np.max(np.abs(new - V)); V = new
            if d < tol: break
        s.V = V.reshape(s.K, s.m); s.n_it = n; return s
    def best(s, Y):
        """V и (t, u) лучшей дуги для точек Y (n,2)."""
        Y = np.atleast_2d(Y); iy, k, q, t, u = s.pairs(Y); J = np.full(len(Y), BIG); T = np.zeros(len(Y)); U = np.zeros(len(Y))
        if len(iy):
            iy, k, q = iy.astype(int), k.astype(int), q.astype(int); j0, a = s.qj0[q], s.qa[q]; V0, V1 = s.V[k, j0], s.V[k, j0 + 1]
            val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - 1e-9)) | ((V1 >= BIG / 2) & (a > 1e-9))] = BIG
            o = np.lexsort((val, iy)); first = o[np.r_[True, iy[o][1:] != iy[o][:-1]]]; J[iy[first]] = val[first]; T[iy[first]] = t[first]; U[iy[first]] = u[first]
        g = (np.abs(wrap(Y[:, 0])) <= RHO + 1e-9) & (np.abs(Y[:, 1]) <= RHO + 1e-9); J[g] = 0.
        return J, T, U
    def rollout(s, Q, mode='dt', dt=.05, tmax=40., h=.01):
        """все старты разом. mode 'dt' — держать u шаг dt и перепланировать; 'arc' — держать u всю дугу."""
        Y = np.array(Q, float); n = len(Y); T = np.full(n, np.inf); t = np.zeros(n); act = np.ones(n, bool); pu = np.full(n, np.nan); sw = np.zeros(n, int)
        hold = np.zeros(n); u = np.zeros(n)
        while act.any() and t[act].min() < tmax:
            need = act & (hold <= 1e-12)
            if need.any():
                J, ta, un = s.best(Y[need]); i = np.flatnonzero(need); dead = J >= BIG / 2; act[i[dead]] = False
                i, ta, un = i[~dead], ta[~dead], un[~dead]; sw[i] += (np.abs(un - pu[i]) > UM) & ~np.isnan(pu[i]); pu[i] = un; u[i] = un
                hold[i] = np.minimum(dt, ta) if mode == 'dt' else ta
            i = np.flatnonzero(act)
            if not len(i): break
            hh = np.minimum(h, hold[i]); x, w = step(Y[i, 0], Y[i, 1], u[i], hh); Y[i, 0], Y[i, 1] = x, w; t[i] += hh; hold[i] -= hh
            g = (np.abs(wrap(x)) <= RHO + 1e-9) & (np.abs(w) <= RHO + 1e-9); T[i[g]] = t[i[g]]; act[i[g]] = False; act[i[t[i] >= tmax]] = False
        return T, sw

def stats(r):
    if not len(r): return None
    return dict(mean=round(float(np.mean(r)), 4), med=round(float(np.median(r)), 4), min=round(float(np.min(r)), 3), max=round(float(np.max(r)), 3))

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; tau = float(os.environ.get('TAU', .5)); r = float(os.environ.get('R', .1)); NQ = int(os.environ.get('NQ', 100))
    t0 = time.time(); B = ButterflyPend(N=N, tau=tau, r=r).build(); tb = time.time() - t0
    t0 = time.time(); B.solve(); print(json.dumps(dict(N=N, tau=tau, r=r, TL=TL, nodes=B.K * B.m, pairs=len(B.e[0]), pairs_per_node=round(len(B.e[0]) / (B.K * B.m), 1), sec_pairs=round(tb), sec_bellman=round(time.time() - t0, 1),
                                                    iters=B.n_it, BIG_nodes=round(float((B.V >= BIG / 2).mean()), 3), BIG_nodes_w2=round(float((B.V >= BIG / 2)[np.abs(B.P[..., 1]) <= 2].mean()), 3))), flush=True)
    rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1)[:NQ]          # те же старты, что exact_switch_pend
    t0 = time.time(); S = Grid(); S.solve((-1., 1.), 0.); Vg = S.Vq(Q[:, 0], Q[:, 1]); Ta = np.array(S.rollout(Q, (-1., 1.), 0.)[0]); print('grid sec', round(time.time() - t0), 'Ta/Vg', stats(Ta / Vg), flush=True)
    Vb = B.best(Q)[0]; ok = (Vg > .05) & (Vg < BIG / 2); m = ok & (Vb < BIG / 2)
    print(json.dumps(dict(V_cover=round(float(m.sum() / ok.sum()), 3), V_over_Vgrid=stats(Vb[m] / Vg[m])), ensure_ascii=False), flush=True)
    tag = '%d_tau%g_r%g%s' % (N, tau, r, '_tl%g' % TL if TL else ''); res = [Vg, Ta, Vb]
    for mode in ('dt', 'arc'):
        t0 = time.time(); T, sw = B.rollout(Q, mode); res += [T, sw]; f = np.isfinite(T) & ok
        print(json.dumps(dict(agent=mode, reach=round(float(f.sum() / ok.sum()), 3), T_over_Vgrid=stats(T[f] / Vg[f]), T_over_Ta=stats(T[f] / Ta[f]), sw_med=float(np.median(sw[f])), sw_mean=round(float(sw[f].mean()), 1),
                              sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save('butterfly_pend_%s.npy' % tag, np.array(res))                                 # строки: Vсетки, Tагента сетки, Vбабочек, T dt, sw dt, T arc, sw arc
