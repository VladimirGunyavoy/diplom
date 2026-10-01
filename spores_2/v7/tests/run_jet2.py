import sys, time, pickle, os; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import di, pend
from src.cells7.cover import cover_layer, cover_tubes
import src.cells7.cell as cm
cm.NT = int(os.environ.get("NT", 32))
from src.cells7.cell import Cell
from src.cells7.value_jet import Jet, goal_chain
name, m, nov = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); S = di() if name == 'di' else pend(); t0 = time.time()
def get(k, seed):
    fn = f'/tmp/cells_{name}_{m}_{k}_{seed}{os.environ.get("TUBES", "")}.pkl'
    if os.path.exists(fn): return [Cell(S, k, c, r=r, tau=tau, tol=1e9) for c, r, tau in pickle.load(open(fn, 'rb'))]
    L = (cover_tubes if os.environ.get('TUBES') else cover_layer)(S, k, m=m, seed=seed)[0]; pickle.dump([(C.c, C.r, C.tau) for C in L], open(fn, 'wb')); return L
layers = [sum([get(k, sd) for sd in range(nov)], []) for k in (0, 1)]; print('cells', [len(L) for L in layers], round(time.time() - t0), flush=True)
def T(x, v): return v + 2 * np.sqrt(x + v * v / 2) if x + v * abs(v) / 2 > 0 else -v + 2 * np.sqrt(v * v / 2 - x)
rng = np.random.default_rng(1); Y = rng.uniform(-.8, .8, (200, 2)); ref = np.array([T(*(y * S.scale)) for y in Y])
for st in [float(a) for a in sys.argv[4:]]:
    J = Jet(S, layers, np.zeros(2), s_tol=st, lat=os.environ.get('LAT','lin'), lam=float(os.environ.get('LAM', 0)), chains=[goal_chain(S, k, np.zeros(2)) for k in (0, 1)]); J.solve(); q = J.query(Y); r = q / ref; f = r[np.isfinite(r)]
    print('s_tol', st, 'nodes finite', round(float(np.isfinite(J.V).mean()), 2), 'sweeps', J.sweeps, 'query finite', len(f) / len(r), 'median', round(float(np.median(f)), 3), 'mean', round(float(f.mean()), 3), 'q10/q90', np.round(np.quantile(f, [.1, .9]), 3), flush=True)
