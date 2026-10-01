"""python3 tests/run_covern.py <система> [NP max_cells tmax] — слой 0 и 1 (первые два), json в reports/cn_<система>.json"""
import sys, time, json; sys.path.insert(0, '.')
from src.cells7.systemsn import SYSTEMS
from src.cells7.covern import cover_layer_n, metrics_n
name = sys.argv[1]; NP, MC, TM = (int(a) for a in sys.argv[2:5]) if len(sys.argv) > 4 else (4000, 300, 600)
S = SYSTEMS[name](); out = {}
for k in (0, 1):
    t0 = time.time(); cells, P, cov, curve = cover_layer_n(S, k, NP, MC, TM, log=lambda *a: print(name, k, a, flush=True))
    M = metrics_n(S, cells, P, cov); M['sec'] = time.time() - t0; M['curve'] = curve; out['layer%d' % k] = M; print(name, k, {a: b for a, b in M.items() if a != 'curve'}, flush=True)
json.dump(out, open('reports/cn_%s.json' % name, 'w'), indent=1)
