import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import di
from src.cells7.cover import cover_layer
from src.cells7.graph import build_graph, solve_V, query
m = int(sys.argv[1]); S = di(); t0 = time.time()
layers = [cover_layer(S, k, m=m)[0] for k in (0, 1)]; print('cover', [len(L) for L in layers], round(time.time() - t0), flush=True)
edges, gt = build_graph(S, layers, np.zeros(2)); V = solve_V(layers, edges, gt); print('graph', sum(len(e) for e in edges.values()), 'goal cells', len(gt), 'V', len(V), round(time.time() - t0), flush=True)
def T(x, v): return v + 2 * np.sqrt(x + v * v / 2) if x + v * abs(v) / 2 > 0 else -v + 2 * np.sqrt(v * v / 2 - x)
rng = np.random.default_rng(1); r = []
for _ in range(40):
    y = rng.uniform(-.8, .8, 2); x, v = y * S.scale; ref = T(x, v); q = query(S, layers, edges, gt, V, y); r.append(q / ref)
r = np.array(r); print('ratio finite', np.isfinite(r).mean(), 'median', np.median(r[np.isfinite(r)]), 'max', r[np.isfinite(r)].max(), 'mean', r[np.isfinite(r)].mean())
