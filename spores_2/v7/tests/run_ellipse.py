"""python3 tests/run_ellipse.py <система> [число запросов N_per_ring]  → reports/ellipse_<система>.json: рост общего пула на подряд идущих запросах"""
import sys, time, json; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systemsn import SYSTEMS
from src.cells7.ellipse import Pool
name = sys.argv[1]; NQ = int(sys.argv[2]); NR = int(sys.argv[3]); S = SYSTEMS[name](); P = Pool(S); rng = np.random.default_rng(5); out = []
for q in range(NQ):
    s, g = S.sample(rng, 2); t0 = time.time(); r = P.query(s, g, N=NR); r['sec'] = round(time.time() - t0, 1); r['d'] = float(np.linalg.norm(S.wrap(s - g))); out.append(r); print(q, r, flush=True)
    json.dump(out, open('reports/ellipse_%s.json' % name, 'w'))
