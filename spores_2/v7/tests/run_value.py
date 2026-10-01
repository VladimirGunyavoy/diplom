import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import di
from src.cells7.cover import cover_layer
from src.cells7.value import Field
m = int(sys.argv[1]); S = di(); t0 = time.time()
layers = [cover_layer(S, k, m=m)[0] for k in (0, 1)]; print('cover', [len(L) for L in layers], round(time.time() - t0), flush=True)
F = Field(S, layers, np.zeros(2)); print('field', round(time.time() - t0), 'goal nodes', F.goal.sum(), flush=True); F.solve(); print('solved sweeps', F.sweeps, 'finite', (F.V < 1e8).mean(), round(time.time() - t0), flush=True)
def T(x, v): return v + 2 * np.sqrt(x + v * v / 2) if x + v * abs(v) / 2 > 0 else -v + 2 * np.sqrt(v * v / 2 - x)
rng = np.random.default_rng(1); r = []
for _ in range(60):
    y = rng.uniform(-.8, .8, 2); x, v = y * S.scale; r.append(F.query(y) / T(x, v))
r = np.array(r); f = r[r < 1e8]; print('finite', len(f) / len(r), 'median', np.median(f), 'mean', f.mean(), 'min', f.min(), 'max', f.max())
