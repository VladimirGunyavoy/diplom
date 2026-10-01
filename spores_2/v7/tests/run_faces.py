import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, '../v5chain/reports/research')
import numpy as np
from src.cells7.systems import di, pend
from src.cells7.faces import Faces, lines_uniform
from v7_faces_di import tstar_box
n = int(sys.argv[1]); m = int(sys.argv[2]) if len(sys.argv) > 2 else 4; S = di(); ln = lines_uniform(n if n % 2 == 0 else n + 1, False)
t0 = time.time(); F = Faces(S, ln, ln, m=m); print('lines', len(ln), 'probes', len(F.P), 'ax', F.ax, round(time.time() - t0, 1), flush=True)
V = F.solve(); print('solved iters', F.iters, 'finite', np.isfinite(V).mean(), round(time.time() - t0, 1), flush=True)
rng = np.random.default_rng(0); Q = rng.uniform(-.375, .375, (500, 2)); T, sw = F.rollout(Q); ph = Q * S.scale
Ts = tstar_box(ph[:, 0], ph[:, 1], F.ax * 4); ok = np.isfinite(T) & (Ts > .05); r = T[ok] / Ts[ok]
print('дошли', np.isfinite(T).mean(), 'T/T* mean %.3f med %.3f max %.2f min %.3f' % (r.mean(), np.median(r), r.max(), r.min()), 'перекл. мед', np.median(sw[ok]), round(time.time() - t0, 1))
