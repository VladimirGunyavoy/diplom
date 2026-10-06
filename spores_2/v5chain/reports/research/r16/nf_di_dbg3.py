import numpy as np, grow_cells2d as G, sys, collections
from scipy.spatial import cKDTree
G.BARRIER = cKDTree(np.load(sys.argv[1])); rng = np.random.default_rng(1)
for u in (-1., 0.): G.build_layer(u, rng)
orig = G.grow2; seeds = []
def g2(p, u, idx, rm, tm):
    c = orig(p, u, idx, rm, tm)
    if c is not None: seeds.append((p.copy(), c))
    return c
G.grow2 = g2
def log(u, cells):
    if len(cells) >= 600:
        ps = np.array([s[0] for s in seeds[-200:]]); k = collections.Counter(map(tuple, ps.round(6)))
        print('last 200 seeds: distinct (1e-6)', len(k), 'top', k.most_common(3), flush=True)
        idx = G.Index()
        for x in cells: idx.add(x)
        p, c = seeds[-1]; print('last seed', p, 'covered by all', idx.covered(p[None]), 'cell nb nf', c.nb, c.nf, 'r', c.r, 'G0', c.G[0].round(4).tolist(), flush=True)
        sys.exit(0)
G.build_layer(1., rng, log); print('layer done')
