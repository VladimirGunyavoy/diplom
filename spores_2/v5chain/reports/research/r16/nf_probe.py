"""research-16: почему NORMFRONT (303/304 worker-b1) дал iters=0 / reach 0 — диагностика V после первого solve."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

k = 0
for c in A.cells:
    nt, m = c.G.shape[:2]; v = A.V[c.o:c.o + nt * m].reshape(nt, m); fin = v < G.BIG / 2
    if not (fin.any() and not fin.all()): continue
    k += 1
    if k > 4: break
    Y = c.G[-1]; print('cell u', c.u, 'nt', nt, 'last row', Y.round(3).tolist(), 'finite', fin[-1].tolist(), 'V', v[-1].round(2).tolist())
    for u in G.US:
        tg = A.tgoal(Y, u); I, IDX, W = A.stencils(G.step(Y, u))
        print('  u', u, 'tgoal', tg.round(3).tolist(), 'img', G.step(Y, u).round(3).tolist())
        for q in range(m):
            sel = I == q
            print('    node', q, 'hits', int(sel.sum()), [(A.V[IDX[t]].round(1).tolist(), W[t].round(2).tolist()) for t in np.flatnonzero(sel)[:3]])
