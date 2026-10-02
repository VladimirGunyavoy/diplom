"""research hub-research-7 (aida): маятник u_max=.3, цель — верх |φ|,|ω|≤.1; цена ∫(1+ρ(u/u_max)²)dt. Полулагранжева сетка (φ периодично, ω∈[-W,W]),
V для наборов U: bang {±1}, tri {−1,0,1}, spec11 (в долях u_max); BIG-конечное, Якоби. Ошибка сетки у всех наборов одна ⇒ отношения V честные.
Плюс агент по каждой V (argmin_u dt·c(u)+V(φ_u), rk4) — реальная цена и число переключений."""
import numpy as np, json, sys
UM, W, BIG = .3, 4.0, 1e3
def f(x, w, u): return w, np.sin(x) + u
def step(x, w, u, h):
    k1 = f(x, w, u); k2 = f(x + h / 2 * k1[0], w + h / 2 * k1[1], u); k3 = f(x + h / 2 * k2[0], w + h / 2 * k2[1], u); k4 = f(x + h * k3[0], w + h * k3[1], u)
    return x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
class Grid:
    def __init__(s, nx=361, nw=321, dt=.03):
        s.gx = np.linspace(-np.pi, np.pi, nx, endpoint=False); s.gw = np.linspace(-W, W, nw); s.nx, s.nw, s.dt = nx, nw, dt
        X, Wv = np.meshgrid(s.gx, s.gw, indexing='ij'); s.X, s.Wv = X.ravel(), Wv.ravel()
    def interp_w(s, x, w):
        hx = 2 * np.pi / s.nx; hw = s.gw[1] - s.gw[0]; fx = (wrap(x) + np.pi) / hx; i = np.floor(fx).astype(int) % s.nx; a = fx - np.floor(fx); i1 = (i + 1) % s.nx
        fw = np.clip((w + W) / hw, 0, s.nw - 1 - 1e-9); j = fw.astype(int); b = fw - j; out = np.abs(w) > W
        return np.stack([i * s.nw + j, i * s.nw + j + 1, i1 * s.nw + j, i1 * s.nw + j + 1], -1), np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], -1), out
    def goal(s, x, w): return (np.abs(wrap(x)) <= .1) & (np.abs(w) <= .1)
    def solve(s, U, rho, tol=1e-7, it=40000):
        F = []
        for u in U:
            x1, w1 = step(s.X, s.Wv, u * UM, s.dt); idx, wt, out = s.interp_w(x1, w1); F.append((s.dt * (1 + rho * u * u), idx, wt, out))
        G = s.goal(s.X, s.Wv); V = np.full(len(s.X), BIG); V[G] = 0
        for n in range(it):
            Vn = np.full(len(s.X), BIG)
            for c, idx, wt, out in F: Vn = np.minimum(Vn, np.where(out, BIG, c + np.sum(wt * V[idx], -1)))
            Vn[G] = 0; d = np.max(np.abs(Vn - V)); V = Vn
            if d < tol and n > 200: break
        s.V = V; s.n = n; return V
    def Vq(s, x, w): idx, wt, out = s.interp_w(x, w); return np.where(out, BIG, np.sum(wt * s.V[idx], -1))
    def rollout(s, Q, U, rho, tmax=60.):
        x, w = Q[:, 0].copy(), Q[:, 1].copy(); n = len(x); J = np.zeros(n); done = s.goal(x, w); sw = np.zeros(n); pu = np.full(n, np.nan); U = np.asarray(U)
        for _ in range(int(tmax / s.dt)):
            a = np.nonzero(~done)[0]
            if not len(a): break
            c = np.stack([s.dt * (1 + rho * u * u) + s.Vq(*step(x[a], w[a], u * UM, s.dt)) for u in U], 1); u = U[np.argmin(c, 1)]
            p = pu[a]; sw[a] += np.isfinite(p) & (np.abs(u - np.where(np.isfinite(p), p, u)) > .5); pu[a] = u
            for _k in range(3):
                h = s.dt / 3; x[a], w[a] = step(x[a], w[a], u * UM, h); J[a] += h * (1 + rho * u * u); g = s.goal(x[a], w[a]); done[a[g]] = True; a, u = a[~g], u[~g]
        J[~done] = np.inf; return J, sw
if __name__ == '__main__':
    rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 200), rng.uniform(-2, 2, 200)], 1)
    S = Grid(); sets = {'bang': (-1., 1.), 'tri': (-1., 0., 1.), 'spec11': tuple(np.linspace(-1, 1, 11))}; res = []
    for rho in (0., .5, 2.):
        R = {}
        for nm, U in sets.items():
            V = S.solve(U, rho); R[nm] = (S.Vq(Q[:, 0], Q[:, 1]), *S.rollout(Q, U, rho), S.n)
        base = R['spec11'][0]
        for nm in sets:
            Vq, J, sw, n = R[nm]; ok = (base < BIG / 2) & (Vq < BIG / 2) & (base > .3); okj = np.isfinite(J) & np.isfinite(R['spec11'][1]) & (base > .3)
            d = dict(rho=rho, U=nm, iters=n, V_over_spec=dict(mean=round(float(np.mean(Vq[ok] / base[ok])), 4), max=round(float(np.max(Vq[ok] / base[ok])), 3)),
                     reach=round(float(np.isfinite(J).mean()), 3), J_over_specJ=round(float(np.mean(J[okj] / R['spec11'][1][okj])), 4), J_over_Vspec=round(float(np.mean(J[okj] / base[okj])), 4), sw_med=float(np.median(sw[okj])))
            print(json.dumps(d), flush=True); res.append(d)
    json.dump(res, open('pend_cost_grid.json', 'w'), indent=1)
