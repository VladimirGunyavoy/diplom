"""research-22, эксп. 13б: сетка (shoot_grid) + доводка SLSQP ТОЛЬКО лучшей топологии (2 старта: равные доли T сетки и T сетки на первой дуге)."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G, fin
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from scipy.optimize import minimize
import finish_gen as FGm
ST = [0, 0.]
def shoot_pol(y, f, US, RHOV, wrapy, tmax, NA=3, inits=2):
    T0, tp = fin.shoot_grid(y, f, US, RHOV, wrapy, tmax, NA)
    if tp is None: return T0, tp
    t0 = time.time(); R = .9 * np.asarray(RHOV); best = (T0, tp); n = len(tp)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FGm.flow(z, US[k], max(dt, 0.), f)
        return wrapy(z)
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if r.success and np.all(np.abs(end(r.x)) <= np.asarray(RHOV)) and r.x.sum() < best[0]: best = (float(r.x.sum()), tp)
    ST[0] += 1; ST[1] += time.time() - t0; return best
