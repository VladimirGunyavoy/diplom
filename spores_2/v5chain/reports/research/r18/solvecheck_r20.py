# research-20: сходимость solve с подклетками (MROW) и без — итераций до max|ΔV| < 1e-3 / 1e-6 / 1e-9; узлы-дубли на границах блоков
import os, sys, numpy as np, time, json
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(0); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []; t = time.time()
for u in G.US: l, ix = G.build_layer(u, rng); A.layers.append(l); A.idx.append(ix)
A.finish(); tb = time.time() - t; t = time.time(); A.solve(); ts = time.time() - t; d = np.array(G.SOLVED)
first = lambda e: int(np.argmax(d < e)) if (d < e).any() else -1
print(json.dumps(dict(mrow=G.MROW, cells=len(A.cells), nodes=int(A.N), iters=len(d), it_1e3=first(1e-3), it_1e6=first(1e-6), it_1e9=first(1e-9), t_build=round(tb, 1), t_solve=round(ts, 1), big=round(float((A.V >= G.BIG / 2).mean()), 4))), flush=True)
np.save(sys.argv[1], d)
B_ = A.V >= G.BIG / 2; rep = {}
for c in A.cells:
    nt, m = c.G.shape[:2]; b = B_[c.o:c.o + nt * m].reshape(nt, m); key = 'm%d_%s' % (m, 'blk' if getattr(c, 'stc', None) else 'full'); r = rep.setdefault(key, [0, 0, 0, 0, 0])
    r[0] += 1; r[1] += b.size; r[2] += int(b.sum()); r[3] += int(b[-1].sum()); r[4] += int(b[:, [0, -1]].sum())
    if getattr(c, 'DEAD', None) is not None: rep.setdefault(key + '_dead', [0])[0] += int(c.DEAD.sum())
print('BIG по типам клеток [клеток, узлов, BIG, BIG в последней строке, BIG в крайних столбцах]:', json.dumps(rep), flush=True)
