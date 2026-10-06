import numpy as np, grow_cells2d as G, sys
def log(u, cells):
    if len(cells) % 25 == 0: print('NF', G.NORMFRONT, len(cells), dict(G.GSTAT), 'r med', np.median([x.r for x in cells[-25:]]).round(3), 'rows', np.median([x.nb + x.nf for x in cells[-25:]]), flush=True)
    if len(cells) >= 150: sys.exit(0)
G.build_layer(1., np.random.default_rng(0), log); print('NF', G.NORMFRONT, 'layer done', dict(G.GSTAT), flush=True)
