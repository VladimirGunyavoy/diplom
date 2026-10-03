import sys, time; sys.path.insert(0, '.'); sys.argv = ['x']
from src.cells7.butterfly_4d import *
S = B4Fast(N=50000).solve(); rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-1, 1, (300, 2)), rng.uniform(-.7, .7, (300, 2))]
Ts = np.maximum(tstar_box(Q[:, 0], Q[:, 2], RHO), tstar_box(Q[:, 1], Q[:, 3], RHO)); ok = Ts > .05; Q, Ts = Q[ok][:150], Ts[ok][:150]
for dt in (.06, .03, .015):
    t0 = time.time(); T, sw = S.rollout(Q, dt=dt); f = np.isfinite(T); r = T[f] / Ts[f]
    print('dt', dt, 'дошли', round(float(f.mean()), 3), 'T/T* mean', r.mean().round(3), 'med', np.median(r).round(3), 'p90', np.percentile(r, 90).round(3), 'max', r.max().round(2), 'выбросов>1.5', int((r > 1.5).sum()), 'sw med', np.median(sw), 'сек', round(time.time() - t0), flush=True)
