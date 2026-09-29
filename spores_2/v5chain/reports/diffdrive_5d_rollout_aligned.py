"""Схема B: rollout5 на согласованном поле (almax=π/4, nth=64, h=.125, n=8, dt=.5). Запуск: python3 reports/diffdrive_5d_rollout_aligned.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import pi
from src.atlas_dd.five_d import make_grid, solve5, rollout5
al = pi / 4
g = make_grid(8, 0.125, 64, wmax=3 * al * 0.5, almax=al, dt=0.5)
T, it = solve5(g, iters=400); mv, mw, n = g['mv'], g['mw'], g['n']
for st in [(0.75, 0, 0, 0, 0), (-0.75, 0.5, 0, 0, 0), (0.5, 0.5, pi / 2, 0, 0), (-0.5, -0.5, pi, 0, 0)]:
    Tq = T[mv, mw, n + int(round(st[0] / 0.125)), n + int(round(st[1] / 0.125)), int(round(st[2] / (2 * pi / 64))) % 64]
    for d in (1, 3):
        c, tr = rollout5(st, T, g, depth=d); e = tr[-1]
        print(f"{st[:3]} depth={d}: T={Tq:.3f} время {len(c)*g['dt']:.2f} конец xy=({e[0]:.3f},{e[1]:.3f}) θ={e[2]:.3f} v={e[3]:.2f} ω={e[4]:.2f}", flush=True)
