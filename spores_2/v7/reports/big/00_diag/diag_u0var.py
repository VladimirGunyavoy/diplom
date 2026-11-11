"""Д1(б): u=0 вне / только в радиусе R от цели (метрика θ,ω), ±.3 целиком; 100 стартов mean/med/p90/max."""
import sys, os, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; A = B.make_atlas(); cu = np.array([list(G.US).index(c.u) for c in A.cells]); dmin = np.array([np.hypot(G.wrap(c.G.reshape(-1, 2)[:, 0]), c.G.reshape(-1, 2)[:, 1]).min() for c in A.cells]); res = {}
V = os.environ.get('V', '')
for name, keep in [x for x in (('без u=0', np.flatnonzero(cu != 1)), ('u=0 в R<=0.7', np.flatnonzero((cu != 1) | (dmin <= .7))), ('u=0 в R<=1.5', np.flatnonzero((cu != 1) | (dmin <= 1.5))), ('u=0 в R<=3', np.flatnonzero((cu != 1) | (dmin <= 3.)))):
    A2 = B.sub(A, keep); ts = B.solve(A2); s, _ = B.eval100(A2); res[name] = dict(cells=len(A2.cells), u0_cells=int((cu[keep] == 1).sum()), nodes=int(A2.N), t_solve=ts, **s); print(name, res[name], flush=True)
    json.dump(res, open(os.environ['BIGOUT'] + '/pend_u0var' + V + '.json', 'w'), indent=1)
