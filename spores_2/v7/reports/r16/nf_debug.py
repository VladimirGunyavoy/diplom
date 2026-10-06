"""research-16: почему NORMFRONT (303/304 worker-b1) дал iters=0 / reach 0 — диагностика V после первого solve."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cells7'))
import grow_cells2d as G
from scipy.spatial import cKDTree
e_ = np.linspace(-G.RHO, G.RHO, 41); GB = np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]]; G.BARRIER = cKDTree(GB)
t0 = time.time(); rng = np.random.default_rng(0); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []
for u in G.US: l, ix = G.build_layer(u, rng); A.layers.append(l); A.idx.append(ix)
A.finish(); print('cells', [len(l) for l in A.layers], 'N', A.N, 'goal nodes', int(A.goal.sum()), 'build s', round(time.time() - t0, 1), 'NFSTAT', G.NFSTAT, flush=True)
Vg = np.full(A.N, np.inf)
for u in G.US: Vg = np.minimum(Vg, A.tgoal(A.P, u))
print('nodes tgoal finite', int(np.isfinite(Vg).sum()), flush=True)
for u in G.US:
    I, IDX, W = A.stencils(G.step(A.P, u)); print('u', u, 'stencil pairs', len(I), 'distinct nodes', len(np.unique(I)), flush=True)
A.solve(); print('iters', A.n_it, 'finite V', int((A.V < G.BIG / 2).sum()), flush=True)
# геометрия: длины строк, разброс шага между строками (вдоль столбцов) в NF-клетках
d = []; bad = 0
for c in A.cells:
    g = c.G; dd = np.linalg.norm(np.diff(g, axis=0), axis=2); d.append(dd.ravel())
    for i in range(len(g) - 1): bad += not G.quad_convex(g[i], g[i + 1])
d = np.concatenate(d); print('row step |dG| pct 1/50/99', np.percentile(d, [1, 50, 99]).round(4), 'nonconvex row pairs', bad, flush=True)
np.savez(os.path.join(os.path.dirname(__file__), 'nf_debug_%d.npz' % G.NORMFRONT), P=A.P, V=A.V, goal=A.goal)
