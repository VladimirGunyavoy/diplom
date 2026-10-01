import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, '../v5chain/reports/research')
import numpy as np
from src.cells7.systems import di
from src.cells7.faces import Faces, lines_graded
from v7_faces_di import tstar_box
S = di(); rng = np.random.default_rng(0); Q = rng.uniform(-.375, .375, (500, 2)); ph = Q * S.scale
for a, ratio in ((.025, 1.0), (.025, 1.1), (.025, 1.2), (.0125, 1.15), (.0125, 1.3), (.0125, 1.5)):
    ln = lines_graded(a, ratio, 200); t0 = time.time(); F = Faces(S, ln, ln, m=4); F.solve(); T, sw = F.rollout(Q)
    Ts = tstar_box(ph[:, 0], ph[:, 1], F.ax * 4); ok = np.isfinite(T) & (Ts > .05); r = T[ok] / Ts[ok]
    print(f'a={a} ratio={ratio}: линий {len(ln)} проб {len(F.P)} дошли {np.isfinite(T).mean():.2f} T/T* mean {r.mean():.3f} max {r.max():.2f} сек {time.time()-t0:.1f}', flush=True)
