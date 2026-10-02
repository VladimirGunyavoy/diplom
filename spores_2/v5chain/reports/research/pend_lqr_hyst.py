"""маятник ρ=0 (+гистерезис eps): агент по сетке V (spec11) + LQR-клетка у верха (u=sat(−Kx) при xᵀPx≤c). Скачки u и время против чистого агента."""
import numpy as np, json
from scipy.linalg import solve_continuous_are
from pend_cost_grid import *
A = np.array([[0, 1.], [1, 0]]); B = np.array([[0], [1.]]); P = solve_continuous_are(A, B, np.eye(2), np.array([[1 / UM**2]])); K = (UM**2 * B.T @ P).ravel()
rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1)
S = Grid(); U = np.linspace(-1, 1, 11); S.solve(tuple(U), 0.); res = []
for c_lqr, eps in ((.5, 0.), (.5, 1e-3), (.5, 3e-3), (.5, 1e-2), (.5, 3e-2)):
    x, w = Q[:, 0].copy(), Q[:, 1].copy(); done = S.goal(x, w); pu = np.full(100, np.nan); T = np.zeros(100); jumps = np.zeros(100); jg = np.zeros(100)
    for _ in range(int(60 / S.dt)):
        a = np.nonzero(~done)[0]
        if not len(a): break
        c = np.stack([S.dt + S.Vq(*step(x[a], w[a], u * UM, S.dt)) for u in U], 1); u = U[np.argmin(c, 1)]
        p0 = pu[a]; ip = np.where(np.isfinite(p0), np.rint((p0 + 1) * 5).astype(int), 0); keep = np.isfinite(p0) & (c[np.arange(len(a)), np.clip(ip, 0, 10)] <= c.min(1) + eps); u = np.where(keep, np.where(np.isfinite(p0), p0, u), u)   # гистерезис: держим прежнее u, если хуже лучшего ≤ eps
        z = np.stack([wrap(x[a]), w[a]], 1); inb = np.einsum('ni,ij,nj->n', z, P, z) <= c_lqr; u = np.where(inb, np.clip(-(z @ K) / UM, -1, 1), u)
        p = pu[a]; j = np.isfinite(p) & (np.abs(u - np.where(np.isfinite(p), p, u)) > .5); jumps[a] += j; jg[a] += j & (np.hypot(z[:, 0], z[:, 1]) < .5); pu[a] = u
        for _k in range(3):
            h = S.dt / 3; x[a], w[a] = step(x[a], w[a], u * UM, h); T[a] += h; g = S.goal(x[a], w[a]); done[a[g]] = True; a, u = a[~g], u[~g]
    T[~done] = np.inf; res.append(dict(c_lqr=c_lqr, eps=eps, reach=float(np.isfinite(T).mean()), T_mean=round(float(np.mean(T[np.isfinite(T)])), 3), jumps_med=float(np.median(jumps)), jumps_near_goal_med=float(np.median(jg))))
    print(json.dumps(res[-1]), flush=True)
