import numpy as np, grow_cells2d as G, sys
from scipy.spatial import cKDTree
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
cells, _ = G.build_layer(float(sys.argv[1]), np.random.default_rng(0))
nt = np.array([len(c.G) for c in cells]); print('NF', G.NORMFRONT, 'u', sys.argv[1], 'cells', len(cells), 'rows med/90', np.median(nt), np.percentile(nt, 90), dict(G.GSTAT), G.NFSTAT, flush=True)
