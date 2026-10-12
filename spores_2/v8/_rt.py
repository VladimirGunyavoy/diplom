import sys, time; sys.path.insert(0,'.')
import numpy as np
import src.algo.ref_di_disk as R
S = R.starts(200, 1, .2); t0 = time.perf_counter(); T0 = R.T_ref(S, .2); print('\nRES default time %.1fs' % (time.perf_counter() - t0))
g0, z0, r0 = R.GRID0, R.ZOOM_N, R.ZOOM_ROUNDS
R.GRID0 = 19201; R.ZOOM_N = 1201; R.ZOOM_ROUNDS = 6
t0 = time.perf_counter(); T1 = R.T_ref(S, .2); print('\nRES fine time %.1fs' % (time.perf_counter() - t0))
d = T0 - T1; print('\nRES default - fine: max %.3e min %.3e' % (d.max(), d.min()))
bad = np.argsort(-d)[:3]; print('\nRES worst', [(tuple(np.round(S[i], 3)), float('%.2e' % d[i])) for i in bad])
for p in ((1,0), (-2,1), (.5,-2)): print('\nRES pt', p, '%.7f' % R.T_ref(p, .2))
