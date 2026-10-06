# research-20: атлас как в compute.py (без CUT) → V на сетке (фон), агент из 100 стартов прогона (agent.npz Q), управление на шаге — по пути; всё в npz
import os, sys, numpy as np, time, pickle
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
RUN = sys.argv[1]; OUT = sys.argv[2]; t0 = time.time()
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(int(os.environ.get('SEED', 0))); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []
for u in G.US: l, ix = G.build_layer(u, rng); A.layers.append(l); A.idx.append(ix)
A.finish(); A.solve(); tb = time.time() - t0; print('атлас+solve %.0f с, клеток %d' % (tb, len(A.cells)), flush=True)
gx = np.linspace(-np.pi, np.pi, 361); gw = np.linspace(-G.WL, G.WL, 281); GX, GW = np.meshgrid(gx, gw, indexing='ij'); VG = A.vstar(np.c_[GX.ravel(), GW.ravel()]).reshape(GX.shape)
Q = np.load(os.path.join(RUN, 'agent.npz'))['Q']; T, sw, P = A.rollout(Q)                     # P: (шаги+1, n, 2)
UA = np.array(G.US); U = np.full(P.shape[:2], np.nan)
for k in range(P.shape[0] - 1):
    mv = np.any(P[k + 1] != P[k], 1)
    if not mv.any(): continue
    d = np.stack([np.linalg.norm(G.wrap(G.step(P[k][mv], u)[:, 0] - P[k + 1][mv, 0])[:, None] * [1, 0] + (G.step(P[k][mv], u) - P[k + 1][mv]) * [0, 1], axis=1) for u in G.US], 1); U[k, mv] = UA[d.argmin(1)]
np.savez(OUT, gx=gx, gw=gw, VG=VG, P=P, U=U, T=T, sw=sw, Q=Q); print('готово %.0f с' % (time.time() - t0), 'дошли', np.isfinite(T).sum(), '/', len(T), flush=True)
