"""BIG0 (поправка research-30): выброс клеток на 100 внутренних стартах (mean/med/max) + K крупнейших."""
import sys, os, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
A = B.make_atlas(); vol = np.array([B.cell_vol(c) for c in A.cells]); order = np.argsort(-vol); n = len(order); res = {}
for K in [n, 187, 125, 63, 40, 30, 25, 20, 16, 12]:
    A2 = B.sub(A, order[:K]); ts = B.solve(A2); s, _ = B.eval100(A2); res[K] = dict(cells=len(A2.cells), nodes=int(A2.N), t_solve=ts, **s); print(K, res[K], flush=True)
    json.dump(res, open(os.environ['BIGOUT'] + '/pend_drop100.json', 'w'), indent=1)
