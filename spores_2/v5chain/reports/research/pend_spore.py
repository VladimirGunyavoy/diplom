"""research hub-research-7: схема клеток v7_spore_di на МАЯТНИКЕ (верх, φ'' = sin φ + u, u = ±.3), чтобы отделить схему от реализации spore_v (T/Ta 1.52).
Клетка слоя u: спора c, отрезок ⟂ полю, клоны — rk4 (мелкий шаг), узлы m × nt с гало 10%. Локатор: s — из инварианта слоя H_u = ω²/2 + cos φ − uφ
(1D Ньютон по клону), t — Ньютон по Эрмиту узлов вдоль траектории (линейно по s между соседними клонами). φ периодично (образ q ближайший к споре).
V* и агент — как в v7_spore_di (V*(узел) = min_k [Δt + V*(φ_k(узел, Δt))], BIG-конечное, без переходов за 0 времени). Эталон — агент по сетке HJB (pend_cost_grid)."""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_spore_di import Cover, BIG
UM = .3
def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
def fld(Q, u): return np.stack([Q[..., 1], np.sin(Q[..., 0]) + u], -1)
def rk4(Q, u, h, n=4):
    Q = np.array(Q, float); h = np.broadcast_to(np.asarray(h, float), Q.shape[:-1])[..., None] / n
    for _ in range(n):
        k1 = fld(Q, u); k2 = fld(Q + h / 2 * k1, u); k3 = fld(Q + h / 2 * k2, u); k4 = fld(Q + h * k3, u); Q = Q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Q

class PendCover(Cover):
    LAYERS = (UM, -UM)
    def __init__(s, tau=.4, r=.1, m=7, nt=9, halo=.1, seeds=8000, W=3.0, seed=0):
        s.tau, s.r, s.m, s.nt, s.h, s.rho = tau, r, m, nt, halo, .1
        s.sgn = np.linspace(-1, 1, m); s.tgn = np.linspace(-halo, 1 + halo, nt); s.dt0 = (s.tgn[1] - s.tgn[0]) * tau
        rng = np.random.default_rng(seed); s.C = np.zeros((0, 5)); s.Y = np.zeros((0, m, nt, 2)); s.F = s.Y.copy(); s.nrm = np.zeros((0, 2))
        for u in (UM, -UM):
            cand = np.stack([rng.uniform(-np.pi, np.pi, seeds), rng.uniform(-W, W, seeds)], 1)
            cand = cand[np.linalg.norm(fld(cand, u), axis=1) >= .05]                  # у равновесия слоя клетка вырождена — пропуск
            for b in range(0, len(cand), 100):                                       # пачками: спора вне ядер клеток своего слоя
                cb = cand[b:b + 100]
                if len(s.C):
                    _, _, ok = s._locate(s.C, cb, core=True); ok &= (s.C[None, :, 2] == u); cb = cb[~ok.any(1)]
                if len(cb): s._add(np.c_[cb, np.full(len(cb), u), np.full(len(cb), r), np.full(len(cb), tau)])
        s.K = len(s.C); s.X = s.Y[..., 0]; s.Vv = s.Y[..., 1]; s.N = s.K * m * nt

    def _add(s, Cn):
        Y, F, n = s._nodes(Cn); s.C = np.r_[s.C, Cn]; s.Y = np.r_[s.Y, Y]; s.F = np.r_[s.F, F]; s.nrm = np.r_[s.nrm, n]

    def _nodes(s, C):
        K = len(C); f0 = np.stack([C[:, 1], np.sin(C[:, 0]) + C[:, 2]], 1); fn = np.linalg.norm(f0, axis=1); n = np.stack([-f0[:, 1], f0[:, 0]], 1) / fn[:, None]
        S = s.sgn[None, :] * (C[:, 3:4] * (1 + s.h)); Y0 = C[:, None, :2] + S[..., None] * n[:, None, :]
        T = s.tgn[None, :] * C[:, 4:5]; Y = np.zeros((K, s.m, s.nt, 2)); u = C[:, 2][:, None]
        cur = rk4(Y0, u, T[:, 0:1] * np.ones((1, s.m)), n=4); Y[:, :, 0] = cur
        for i in range(1, s.nt): cur = rk4(cur, u, (T[:, i:i + 1] - T[:, i - 1:i]) * np.ones((1, s.m)), n=6); Y[:, :, i] = cur
        return Y, fld(Y, u[:, :, None]), n

    def _locate(s, A, Q, core=False):
        """A — это s.C (все клетки); Q (n,2). S, T (n,K), ok."""
        C = s.C; n, K = len(Q), len(C); x0, w0, u = C[None, :, 0], C[None, :, 1], C[None, :, 2]
        xq = x0 + wrap(Q[:, 0, None] - x0); wq = Q[:, 1, None]                    # образ q у споры
        H = lambda x, w: w * w / 2 + np.cos(x) - u * x
        Hq = H(xq, wq); nx, nw = s.nrm[None, :, 0], s.nrm[None, :, 1]; fn = np.hypot(w0, np.sin(x0) + u)
        S = (Hq - H(x0, w0)) / fn
        for _ in range(4):
            xs, ws = x0 + S * nx, w0 + S * nw; g = H(xs, ws) - Hq; dg = (-np.sin(xs) - u) * nx + ws * nw
            S = S - g / np.where(np.abs(dg) > 1e-9, dg, 1e-9)
        rr, tt = C[None, :, 3], C[None, :, 4]; T = np.full((n, K), np.nan)
        cand = np.abs(S) <= rr * (1 + s.h) + 1e-7
        qi, k = np.nonzero(cand)
        if len(qi):
            sv = S[qi, k]; fs = (sv / (rr[0, k] * (1 + s.h)) + 1) / (s.sgn[1] - s.sgn[0]); j = np.clip(np.floor(fs).astype(int), 0, s.m - 2); a = np.clip(fs - j, 0, 1)
            P = (1 - a)[:, None, None] * s.Y[k, j] + a[:, None, None] * s.Y[k, j + 1]; D = (1 - a)[:, None, None] * s.F[k, j] + a[:, None, None] * s.F[k, j + 1]   # (p, nt, 2)
            q = np.stack([xq[qi, k], wq[qi, 0]], 1); hti = (s.tgn[1] - s.tgn[0]) * tt[0, k]
            i0 = np.argmin(np.sum((P - q[:, None, :])**2, -1), 1); tq = s.tgn[0] * tt[0, k] + i0 * hti
            for _ in range(5):                                                     # Ньютон по Эрмиту
                fi = (tq - s.tgn[0] * tt[0, k]) / hti; i = np.clip(np.floor(fi).astype(int), 0, s.nt - 2); z = np.clip(fi - i, -.5, 1.5)[:, None]
                ar = np.arange(len(qi)); p0, p1, d0, d1 = P[ar, i], P[ar, i + 1], D[ar, i] * hti[:, None], D[ar, i + 1] * hti[:, None]
                h00, h10, h01, h11 = 2 * z**3 - 3 * z**2 + 1, z**3 - 2 * z**2 + z, -2 * z**3 + 3 * z**2, z**3 - z**2
                X = h00 * p0 + h10 * d0 + h01 * p1 + h11 * d1; dX = ((6 * z**2 - 6 * z) * p0 + (3 * z**2 - 4 * z + 1) * d0 + (-6 * z**2 + 6 * z) * p1 + (3 * z**2 - 2 * z) * d1) / hti[:, None]
                tq = tq + np.sum((q - X) * dX, 1) / np.maximum(np.sum(dX * dX, 1), 1e-12)
            res = np.linalg.norm(q - X, axis=1); T[qi, k] = np.where(res < 1e-3, tq, np.nan)
        if core: ok = (np.abs(S) <= rr) & (T >= 0) & (T <= tt)
        else: ok = cand & (T >= -s.h * tt - 1e-7) & (T <= (1 + s.h) * tt + 1e-7)
        return S, np.nan_to_num(T, nan=-1e9), ok & np.isfinite(S)

    @staticmethod
    def phi(Q, u, h): Q2 = rk4(Q, u, h, n=3); Q2[:, 0] = wrap(Q2[:, 0]); return Q2
    def goal(s, x, v): return (np.abs(wrap(x)) <= s.rho) & (np.abs(v) <= s.rho)

    def rollout(s, Q0, dt=None, tmax=40.0, eps=0.0):
        dt = dt or s.dt0; q = np.array(Q0, float); n = len(q); T = np.zeros(n); done = s.goal(q[:, 0], q[:, 1]); dead = np.zeros(n, bool); sw = np.zeros(n); pu = np.zeros(n)
        for _ in range(int(tmax / dt)):
            act = np.nonzero(~done & ~dead)[0]
            if not len(act): break
            c = np.stack([s.Vstar(s.phi(q[act], u, dt)) for u in (UM, -UM)], 1); best = np.min(c, 1); u = np.where(c[:, 0] <= c[:, 1], UM, -UM)
            nod = ~np.isfinite(best); dead[act[nod]] = True; act, u = act[~nod], u[~nod]
            sw[act] += (pu[act] != 0) & (pu[act] != u); pu[act] = u
            for _s in range(4):
                q[act] = _mv(q[act], u, dt / 4); T[act] += dt / 4
                g = s.goal(q[act, 0], q[act, 1]); done[act[g]] = True; act, u = act[~g], u[~g]
        T[~done] = np.inf; return T, sw

def _mv(Q, u, h): Q2 = rk4(Q, u, h, n=2); Q2[:, 0] = wrap(Q2[:, 0]); return Q2

if __name__ == '__main__':
    rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1)
    from pend_cost_grid import Grid
    G = Grid(); G.solve((-1., 1.), 0.); Ta, _ = G.rollout(Q, (-1., 1.), 0.)                               # эталон: агент по мелкой сетке (реальное время)
    for a in sys.argv[1:] or ('.4,.1,8000',):
        tau, r, sd = map(float, a.split(',')); t0 = time.time(); Cv = PendCover(tau=tau, r=r, seeds=int(sd)); tc = time.time() - t0
        t0 = time.time(); V = Cv.solve(); ts = time.time() - t0; Vs = Cv.Vstar(Q)
        T, sw = Cv.rollout(Q); ok = np.isfinite(T) & np.isfinite(Ta) & (Ta > .3)
        print(json.dumps(dict(tau=tau, r=r, cells=int(Cv.K), nodes=int(Cv.N), iters=Cv.iters, finV=round(float((V < BIG / 2).mean()), 3), reach=round(float(np.isfinite(T).mean()), 3),
              T_over_gridagent=dict(mean=round(float(np.mean(T[ok] / Ta[ok])), 4), med=round(float(np.median(T[ok] / Ta[ok])), 4), max=round(float(np.max(T[ok] / Ta[ok])), 3)),
              Vstart_over_gridagent_med=round(float(np.median(Vs[ok] / Ta[ok])), 3), sw_med=float(np.median(sw[ok])), sec=dict(cover=round(tc), solve=round(ts)))), flush=True)
