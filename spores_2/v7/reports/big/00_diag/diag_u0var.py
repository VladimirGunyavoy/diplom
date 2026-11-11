"""Д1(б): u=0 вне / только в радиусе R от цели (метрика θ,ω), ±.3 целиком; 100 стартов mean/med/p90/max."""
import sys, os, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; A = B.make_atlas(); cu = np.array([list(G.US).index(c.u) for c in A.cells]); dmin = np.array([np.hypot(G.wrap(c.G.reshape(-1, 2)[:, 0]), c.G.reshape(-1, 2)[:, 1]).min() for c in A.cells]); res = {}
V = os.environ.get('V', ''); VAR = {'none': lambda: np.flatnonzero(cu != 1)}
for r_ in (0.4, 0.7, 1, 1.5, 3): VAR[str(r_)] = (lambda r_=r_: np.flatnonzero((cu != 1) | (dmin <= r_)))
for name in ([V] if V else list(VAR)):
    keep = VAR[name]()
    A2 = B.sub(A, keep); ts = B.solve(A2); s, _ = B.eval100(A2); res[name] = dict(cells=len(A2.cells), u0_cells=int((cu[keep] == 1).sum()), nodes=int(A2.N), t_solve=ts, **s); print(name, res[name], flush=True)
    json.dump(res, open(os.environ['BIGOUT'] + '/pend_u0var' + V + '.json', 'w'), indent=1)
