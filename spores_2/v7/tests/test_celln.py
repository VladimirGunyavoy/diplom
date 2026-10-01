import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import pend
from src.cells7.celln import SysN, CellN
P2 = pend(0.3); S = SysN('pend', 2, 2, lambda y, k, t: P2.rk4(y, k, t), ((-1, -1), (1, 1)), per=(2.0, 0))
rng = np.random.default_rng(0); t0 = time.time()
for c in [(0.2, 0.1), (-0.5, 0.6), (0.9, -0.3)]:
    C = CellN(S, 0, c); print('cell', c, 'r %.3f tau %.3f err %.1e' % (C.r, C.tau, C.err), round(time.time() - t0, 2))
    Z = C.kernel_samples(rng); d, t, ins = C.locate(Z, 1.0); print(' kernel samples inside', ins.mean())
    Y = rng.uniform(-1, 1, (4000, 2)); print(' random in kernel/halo', C.locate(Y, 1.0)[2].mean(), C.locate(Y, 1.1)[2].mean())
