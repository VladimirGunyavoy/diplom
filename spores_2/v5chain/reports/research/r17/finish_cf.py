"""r17: агент grow3 + финиш стрельбой <= NA дуг (вершины ромба) при V* <= VF; дуги дд — замкнутой формой; длительности SLSQP, конец в коробке цели."""
import sys, os, pickle, itertools, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.expanduser("~/spore_v5/r17/v7/src/cells7")); import grow3 as G; import __main__; __main__.Cell = G.Cell; __main__.HexIdx = G.HexIdx
VF = float(os.environ.get("VF", .6)); NA = int(os.environ.get("NA", 3)); R = G.RHO * .9
def arc(y, u, t):
    x, yy, th = y; v, w = u
    if w == 0: return np.array([x + v * t * np.cos(th), yy + v * t * np.sin(th), th])
    return np.array([x, yy, th + w * t])
TOPS = [tp for n in range(1, NA + 1) for tp in itertools.product(range(4), repeat=n) if all(tp[i] != tp[i + 1] for i in range(n - 1))]
def shoot(y, tmax):
    best = (np.inf, None)
    for tp in TOPS:
        def end(d):
            z = y.copy()
            for k, dt in zip(tp, d): z = arc(z, G.US[k], dt)
            return np.r_[z[:2], G.wrap(z[2])]
        cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
        for d0 in (np.full(len(tp), tmax / (2 * len(tp))), np.full(len(tp), tmax / len(tp))):
            r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * len(tp), constraints=cons, options=dict(maxiter=100, ftol=1e-9))
            if r.success and np.all(np.abs(end(r.x)) <= G.RHO) and r.x.sum() < best[0]: best = (r.x.sum(), tp)
    return best
