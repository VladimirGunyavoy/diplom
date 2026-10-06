"""r17: эталон дд В КОРОБКУ цели |x|,|y|,|th| <= RHO: стрельба <= 5 дуг (вершины ромба), min с точечным эталоном dd_ref_60 (верхняя оценка)."""
import sys, os, numpy as np, itertools
from multiprocessing import Pool
sys.path.insert(0, os.path.expanduser("~/spore_v5/r17/diag")); os.environ.setdefault("NA", "5")
from scipy.optimize import minimize
sys.path.insert(0, os.path.expanduser("~/spore_v5/r17/v7/src/cells7")); import grow3 as G
def arc(y, u, t):
    x, yy, th = y; v, w = u
    return np.array([x + v * t * np.cos(th), yy + v * t * np.sin(th), th]) if w == 0 else np.array([x, yy, th + w * t])
TOPS = [tp for n in range(1, 6) for tp in itertools.product(range(4), repeat=n) if all(tp[i] != tp[i + 1] for i in range(n - 1)) and all((G.US[tp[i]][1] == 0) != (G.US[tp[i + 1]][1] == 0) for i in range(n - 1))]
def one(a):
    y, tmax = a; best = tmax; rng = np.random.default_rng(0)
    for tp in TOPS:
        def end(d):
            z = y.copy()
            for k, dt in zip(tp, d): z = arc(z, G.US[k], dt)
            return np.r_[z[:2], G.wrap(z[2])]
        cons = [{"type": "ineq", "fun": lambda d: G.RHO * .999 - np.abs(end(d))}]
        for _ in range(6):
            d0 = rng.uniform(0, 1, len(tp)); d0 *= rng.uniform(.5, 1.) * tmax / d0.sum()
            r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * len(tp), constraints=cons, options=dict(maxiter=200, ftol=1e-10))
            if r.success and np.all(np.abs(end(r.x)) <= G.RHO + 1e-9) and r.x.sum() < best: best = r.x.sum()
    return best
Q, ref = G.starts_ref()
with Pool(20) as p: rb = np.array(p.map(one, [(G.wrapy(q), r) for q, r in zip(Q, ref)]))
np.save(os.path.expanduser("~/spore_v5/r17/out/dd_refbox_60.npy"), rb); print("refbox/ref mean %.4f min %.4f max %.4f" % ((rb / ref).mean(), (rb / ref).min(), (rb / ref).max()))
