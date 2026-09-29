"""Схема B: rollout5 с притяжением к узлу (snap) на сохранённом поле T5_n16.npy (aida). Запуск: python3 reports/diffdrive_5d_snap.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import pi
from src.atlas_dd.five_d import make_grid, rollout5
al = pi / 4; n = 16
g = make_grid(n, 0.125, 64, wmax=3 * al * 0.5, almax=al, dt=0.5)
T = np.load("T5_n16.npy")
for st in [(0.75, 0, 0, 0, 0), (-0.75, 0.5, 0, 0, 0), (1.5, -1.0, pi / 2, 0, 0), (-1.5, 1.0, pi, 0, 0), (2.0, 2.0, -pi / 2, 0, 0), (-1.0, -1.5, 0.0, 0, 0)]:
    for sn in (0.0, 0.1):
        c, tr = rollout5(st, T, g, depth=3, max_steps=60, snap=sn); e = tr[-1]
        print(f"{st[:3]} snap={sn}: время {len(c)*g['dt']:.1f} конец xy=({e[0]:.3f},{e[1]:.3f}) θ={e[2]:.3f} v={e[3]:.2f} ω={e[4]:.2f}", flush=True)
