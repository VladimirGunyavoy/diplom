"""Схема B на большом поле, согласованная решётка (almax=π/4, dt=.5, h=.125, nth=64): n=16 ([-2,2]²) и n=24 ([-3,3]²) параллельно; сохраняет T в npz и печатает T из покоя + rollout5 depth=3.
Запуск: python3 reports/diffdrive_5d_big.py <n>   (считалось на aida, ~/spore_v5)"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import pi
from src.atlas_dd.five_d import make_grid, solve5, rollout5
n = int(sys.argv[1]); al = pi / 4
g = make_grid(n, 0.125, 64, wmax=3 * al * 0.5, almax=al, dt=0.5)
t0 = time.time(); T, it = solve5(g, iters=600); mv, mw = g['mv'], g['mw']
print(f"n={n} it={it} {time.time()-t0:.0f}s", flush=True)
np.save(f"T5_n{n}.npy", T)
for st in [(0.75, 0, 0, 0, 0), (-0.75, 0.5, 0, 0, 0), (1.5, -1.0, pi / 2, 0, 0), (-1.5, 1.0, pi, 0, 0), (2.0, 2.0, -pi / 2, 0, 0)]:
    ix, iy = n + int(round(st[0] / 0.125)), n + int(round(st[1] / 0.125))
    if not (0 <= ix <= 2 * n and 0 <= iy <= 2 * n): continue
    Tq = T[mv, mw, ix, iy, int(round(st[2] / (2 * pi / 64))) % 64]
    c, tr = rollout5(st, T, g, depth=3, max_steps=60); e = tr[-1]
    print(f"n={n} {st[:3]}: T={Tq:.2f} время {len(c)*g['dt']:.1f} конец xy=({e[0]:.3f},{e[1]:.3f}) θ={e[2]:.3f} v={e[3]:.2f} ω={e[4]:.2f}", flush=True)
