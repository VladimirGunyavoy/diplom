"""Схема B: rollout (глубина перебора 1..3) на поле h=0.125, n=12, dt=0.5. Запуск: python3 reports/diffdrive_5d_rollout.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import pi
from src.atlas_dd.five_d import make_grid, solve5, rollout5
g = make_grid(12, 0.125, 16, dt=0.5)
T, it = solve5(g, iters=400); mv, mw, n = g['mv'], g['mw'], g['n']
for st in [(1.0, 0.0, 0.0, 0, 0), (-1.0, 0.5, 0.0, 0, 0), (0.5, 0.5, pi / 2, 0, 0), (-0.75, -0.75, pi, 0, 0)]:
    Tq = T[mv, mw, n + int(round(st[0] / g['h'])), n + int(round(st[1] / g['h'])), int(round(st[2] / (2 * pi / 16))) % 16]
    for d in (1, 2, 3):
        c, tr = rollout5(st, T, g, depth=d)
        e = tr[-1]
        print(f"старт {st[:3]} depth={d}: T={Tq:.3f}, {len(c)*g['dt']:.2f} с, конец x={e[0]:.3f} y={e[1]:.3f} θ={e[2]:.3f} v={e[3]:.2f} ω={e[4]:.2f}", flush=True)
