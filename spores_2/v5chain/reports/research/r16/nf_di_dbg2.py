import numpy as np, grow_cells2d as G, sys
from scipy.spatial import cKDTree; G.BARRIER = cKDTree(np.load(sys.argv[1])) if len(sys.argv) > 1 else None; rng = np.random.default_rng(1)
for u in (-1., 0.): G.build_layer(u, rng)
def log(u, cells):
    if len(cells) >= 700:
        c = cells[-1]; idx = G.Index()
        for x in cells: idx.add(x)
        print('last cell c', c.c.round(3), 'p0', None if c.p0 is None else c.p0.round(3), 'nb,nf', c.nb, c.nf, 'r', round(c.r, 4), 'off', round(c.off, 4), 'G', c.G.shape, flush=True)
        print('covered p0 (fresh idx)', idx.covered(c.p0[None]), 'covered c', idx.covered(c.c[None]), flush=True)
        print('G rows', c.G.round(3).tolist(), 'DEAD', c.DEAD.astype(int).tolist(), flush=True); sys.exit(0)
G.build_layer(1., rng, log); print('layer done')
