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
d = pickle.load(open(sys.argv[1], "rb")); A = G.Atlas.__new__(G.Atlas); A.layers = d["layers"]; A.finish(); A.V = d["V"]; Q, ref = d["Q"], d["ref"]
Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); fin = np.zeros(n, bool); Tf = np.zeros(n)
for it in range(400):
    act = ~done
    if not act.any(): break
    v = A.vstar(Y)
    for i in np.flatnonzero(act & (v <= VF)):
        tf, tp = shoot(Y[i], 2.5 * max(v[i], .2))
        if tp is not None: T[i] += tf; Tf[i] = tf; done[i] = True; fin[i] = True
    act = ~done
    if not act.any(): break
    tgs = np.stack([A.tgoal(Y, u) for u in G.US], 1); J = np.minimum(tgs, G.DTN + np.stack([A.vstar(G.step(Y, u)) for u in G.US], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
    stuck = J.min(1) >= G.BIG / 2
    for ki, u in enumerate(G.US):
        m = act & ~stuck & (k == ki)
        if m.any():
            h = np.where(np.isfinite(tg[m]), tg[m], G.DTN); Y[m] = G.wrapy(np.array([G.rk4(y[None], u, hh / 8, 8)[0] for y, hh in zip(Y[m], h)]))
            T[m] += h
    done |= G.ingoal(Y) | (act & stuck); T[act & stuck] = np.inf
T[~(G.ingoal(Y) | fin)] = np.inf; f = np.isfinite(T); r = T[f] / ref[f]
print(json.dumps(dict(atlas=sys.argv[1], VF=VF, NA=NA, reach=round(float(f.mean()), 3), fin=int(fin.sum()), mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3), Tf_med=round(float(np.median(Tf[fin])), 3) if fin.any() else None)), flush=True)
np.save(sys.argv[1].replace(".pkl", "_fin%.1f.npy" % VF), T)
