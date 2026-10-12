import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
X = S.starts(g, 100, 1); Tr = R.T_ref(X, G.rho); tm = 3 * Tr + 2
for pol in ('commit', 'look'):
    t0 = time.perf_counter(); T0, _ = G.run_agent(X, tm, **S.POLICIES[pol]); t1 = time.perf_counter(); dl = []; T1, _ = G.run_agent(X, tm, dlog=dl, **S.POLICIES[pol]); t2 = time.perf_counter()
    print('RES', pol, 'rows', len(dl), 'switched', sum(r[10] for r in dl), 'T равны', np.array_equal(T0, T1, equal_nan=True), 'выкл %.2f вкл %.2f' % (t1 - t0, t2 - t1), dl[:2] if dl else '')
