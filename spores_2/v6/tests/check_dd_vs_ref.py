import sys, importlib.util; sys.path.insert(0, '.')
import numpy as np
sp = importlib.util.spec_from_file_location('ref', '../v5chain/reports/research/dd_rhombus_ref.py'); ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
from src.atlas6.dd_atlas import solve_dd, V_at
A = solve_dd(h=0.25); rng = np.random.default_rng(3); rows = []
for _ in range(40):
    p = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)])
    r = min(ref.tgt(*p), ref.tgtgt(*p)[0] if isinstance(ref.tgtgt(*p), tuple) else ref.tgtgt(*p)); rows.append((V_at(A, p), r))
a = np.array(rows); d = a[:, 0] / a[:, 1]
print('V/эталон: mean %.3f min %.3f max %.3f' % (d.mean(), d.min(), d.max())); print(np.round(np.sort(d)[[0, 10, 20, 30, 39]], 3))
