import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.pend import PendAtlas
for u in (0.5, 0.3):
    t0 = time.time(); P = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=0.25, umax=u, R_goal=0.5); it = P.solve(iters=1500)
    print(f"u={u}: мелкая сетка {it} итераций {time.time()-t0:.0f} с, V(низ)={P.value((0.0,0.0)):.3f}, V(-1,1)={P.value((-1.0,1.0)):.3f}, V(2,-1)={P.value((2.0,-1.0)):.3f}, V(.5,0)={P.value((0.5,0.0)):.3f}", flush=True)
