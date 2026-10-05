"""research-14: почему STEPS=1 на зерне 1 даёт 99% недостижимых узлов — цель, рёбра, m/kt."""
import numpy as np, os
import grow_cells2d as G
A = G.Atlas(seed=int(os.environ.get('SEED', 1))); print('клеток', len(A.cells), 'узлов', A.N, 'в цели', int(A.goal.sum()), 'm', np.bincount([c.m for c in A.cells]).tolist(), 'kt', np.bincount([c.kt for c in A.cells]).tolist(), flush=True)
Vg = np.full(A.N, np.inf)
for u in G.US: Vg = np.minimum(Vg, A.tgoal(A.P, u))
print('узлов с прямым попаданием в цель', int(np.isfinite(Vg).sum()), flush=True)
for u in G.US:
    I, IDX, W = A.stencils(G.step(A.P, u)); print('u', u, 'узлов с ребром', len(np.unique(I)), 'из', A.N, flush=True)
bad = [c for c in A.cells if c.G.shape[0] * c.m != (c.G.shape[0]) * c.G.shape[1]]; print('клеток с m != G.shape[1]:', len(bad))
