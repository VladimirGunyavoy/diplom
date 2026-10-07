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
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); r = np.load('di4_ref_60_rho35.npy'); Q = r[:, :4]; TR = r[:, 6]; A.V = np.load(os.environ['VFILE'])
stop = np.c_[Q[:, 0] + Q[:, 2] * abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * abs(Q[:, 3]) / 2]; inf_ = np.abs(stop).max(1) <= 2.5; A.vstar(Q); G.FG.shoot = shoot_pol
t0 = time.time(); T, sw, path = A.rollout(Q); dt = time.time() - t0; ok = np.isfinite(T); q = T / TR; m = ok & inf_
print('VF %s сетка H %.3f + доводка | в поле %d/%d | T/эт: мед. %.3f mean %.3f p90 %.3f max %.2f | T < эталона у %d | %.1f с на 60 (%.0f мс/запрос) | сетка %.0f мс + доводка %.0f мс на стрельбу' % (os.environ.get('VF'), fin.H, m.sum(), inf_.sum(), np.median(q[m]), q[m].mean(), np.percentile(q[m], 90), q[m].max(), (T[m] < TR[m] - 1e-6).sum(), dt, 1e3 * dt / 60, 1e3 * fin.NCALL[1] / max(fin.NCALL[0], 1), 1e3 * ST[1] / max(ST[0], 1)), flush=True)
