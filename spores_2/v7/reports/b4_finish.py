"""4D ДИ: бабочки + финиш классической обратной связью (hub-worker-14). При V ≤ VF оси независимо: u_i = −sign(x_i + v_i|v_i|/2) (кривая переключения к началу координат), fine dt .005 до коробки ±RHO."""
import sys, time; sys.path.insert(0, '.'); sys.argv = ['x']
from src.cells7.butterfly_4d import *
S = B4Fast(N=50000).solve(); rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-1, 1, (300, 2)), rng.uniform(-.7, .7, (300, 2))]
Ts = np.maximum(tstar_box(Q[:, 0], Q[:, 2], RHO), tstar_box(Q[:, 1], Q[:, 3], RHO)); ok = Ts > .05; Q, Ts = Q[ok][:150], Ts[ok][:150]
def fin(y, h=.005, tmax=8.):
    y = y.copy(); t = 0.
    while t < tmax:
        if (np.abs(y) <= RHO + 1e-9).all(): return t
        for i in (0, 1):
            x, v = y[i], y[2 + i]; s = x + v * abs(v) / 2; u = -np.sign(s) if abs(s) > 1e-3 else -np.clip(v / h, -1, 1)
            y[i] += v * h + u * h * h / 2; y[2 + i] += u * h
        t += h
    return np.inf
def run(q, VF, dt=.06):
    y = q.copy(); t = 0.
    while t < 25.:
        if (np.abs(y) <= RHO + 1e-9).all(): return t
        V, A, _ = S.best(y[None])
        if V[0] <= VF: return t + fin(y)
        if not np.isfinite(V[0]) or V[0] >= BIG / 2: return np.inf
        a = A[0]
        for _ in range(6):
            h = dt / 6; y[:2] += y[2:] * h + a * h * h / 2; y[2:] += a * h; t += h
            if (np.abs(y) <= RHO + 1e-9).all(): return t
    return np.inf
for VF in (0., .5, 1., 2.):
    t0 = time.time(); T = np.array([run(q, VF) for q in Q]); f = np.isfinite(T); r = T[f] / Ts[f]
    print('VF', VF, 'дошли', round(float(f.mean()), 3), 'T/T* mean', r.mean().round(3), 'med', np.median(r).round(3), 'p90', np.percentile(r, 90).round(3), 'max', r.max().round(2), 'выбросов>1.5', int((r > 1.5).sum()), 'сек', round(time.time() - t0), flush=True)
