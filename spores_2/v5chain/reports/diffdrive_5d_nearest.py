"""Схема B без интерполяции: конец ребра округляется до узла (g['nearest']). Сравнение T из покоя с интерполяцией."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas_dd.five_d import make_grid, solve5
for h, nth in ((0.25, 16), (0.125, 16)):
    for near in (False, True):
        g = make_grid(int(round(3 / h)), h, nth, dt=0.5); g['nearest'] = near
        t0 = time.time(); T, it = solve5(g)
        n, mv, mw = g['n'], g['mv'], g['mw']
        q = lambda x, y, k: T[mv, mw, n + int(round(x / h)), n + int(round(y / h)), k]
        print(f"h={h} nth={nth} nearest={near}: T(1,0,0)={q(1,0,0):.3f} T(0,1,0)={q(0,1,nth//4):.3f} T(0,1,θ0)={q(0,1,0):.3f} it={it} {time.time()-t0:.0f}s", flush=True)
