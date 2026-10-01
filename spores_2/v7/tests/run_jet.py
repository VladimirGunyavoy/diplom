import sys, time, pickle, os; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import di, pend
from src.cells7.cover import cover_layer
from src.cells7.value_jet import Jet, goal_chain
name, m = sys.argv[1], int(sys.argv[2]); S = di() if name == 'di' else pend(); t0 = time.time()
fn = f'/tmp/cells_{name}_{m}.pkl'
from src.cells7.cell import Cell
fn = f'/tmp/cells_{name}_{m}.pkl'
if os.path.exists(fn): layers = [[Cell(S, k, c, r=r, tau=tau, tol=1e9) for c, r, tau in L] for k, L in enumerate(pickle.load(open(fn, 'rb')))]
else:
    layers = [cover_layer(S, k, m=m)[0] for k in (0, 1)]; pickle.dump([[(C.c, C.r, C.tau) for C in L] for L in layers], open(fn, 'wb'))
print('cover', [len(L) for L in layers], round(time.time() - t0), flush=True)
def T(x, v): return v + 2 * np.sqrt(x + v * v / 2) if x + v * abs(v) / 2 > 0 else -v + 2 * np.sqrt(v * v / 2 - x)
for st in [float(a) for a in sys.argv[3:]] or [.3]:
    print('s_tol', st)
    J = Jet(S, layers, np.zeros(2), s_tol=st, chains=[goal_chain(S, k, np.zeros(2)) for k in (0, 1)]); print('jet nodes', J.N, 'edges', len(J.src), 'seeds', J.nseed, round(time.time() - t0), flush=True)
    J.solve(); print('sweeps', J.sweeps, 'finite', np.isfinite(J.V).mean(), round(time.time() - t0), flush=True)
    rng = np.random.default_rng(1); Y = rng.uniform(-.8, .8, (200, 2)); q = J.query(Y)
    ref = np.array([T(*(y * S.scale)) for y in Y]) if name == 'di' else None
    r = q / ref; f = r[np.isfinite(r)]; print('finite', len(f) / len(r), 'median', np.median(f), 'mean', f.mean(), 'min', f.min(), 'max', f.max(), 'q10/q90', np.quantile(f, .1), np.quantile(f, .9))
