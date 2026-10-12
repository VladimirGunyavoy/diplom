"""b3: независимая проверка эталона pend: OCP CasADi (хабовый ocp_casadi.solve) на выбранных стартах бенча (100 стартов rng 0). python ocp_check.py 50 28 65 ..."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench'))
import numpy as np, common, ocp_casadi as O
from multiprocessing import Pool
e0 = common.make('pend'); 
def work(i):
    e = common.Pend(nst=100); t0 = time.perf_counter(); b, ns = O.solve(e, e.X0[i], 100 + i, N=int(os.environ.get('NI', 60)), T0s=tuple(float(x) for x in os.environ.get('T0S', '3,6,10').split(','))); return i, b, e.Tref[i], time.perf_counter() - t0
if __name__ == '__main__':
    ids = [int(a) for a in sys.argv[1:]]
    with Pool(min(len(ids), 16)) as p:
        for i, b, r, t in p.imap_unordered(work, ids): print('RES старт %d OCP T=%s T_ref=%.3f (%.0f с)' % (i, 'None' if b is None else '%.3f' % b, r, t), flush=True)
