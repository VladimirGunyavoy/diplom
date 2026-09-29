"""Схема B: rollout5_best на граничных стартах (края и углы поля n=16, [−2,2]², покой, привязка к достижимому узлу; θ ∈ 8 значений). Запуск: python3 reports/diffdrive_5d_edge.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import pi
from src.atlas_dd.five_d import make_grid, rollout5_best, nearest_reachable
al = pi / 4; n = 16
g = make_grid(n, 0.125, 64, wmax=3 * al * 0.5, almax=al, dt=0.5)
T = np.load("T5_n16.npy"); mv, mw = g['mv'], g['mw']
pts = [(-1.75, -1.75), (1.75, -1.75), (-1.75, 1.75), (1.75, 1.75), (0, 1.75), (0, -1.75), (1.75, 0), (-1.75, 0)]
ok = tot = 0; ratios = []; bad = []
for (x, y) in pts:
    for k in range(0, 64, 8):
        st0 = (x, y, k * 2 * pi / 64, 0, 0)
        st0, _, _ = nearest_reachable(st0, T, g)
        Tq = T[mv, mw, n + int(round(st0[0] / 0.125)), n + int(round(st0[1] / 0.125)), int(round(st0[2] / (2 * pi / 64))) % 64]
        if Tq >= 100: continue
        tot += 1
        c, tr = rollout5_best(st0, T, g, max_steps=80, snap=0.25); e = tr[-1]
        d = (e[2] + pi) % (2 * pi) - pi
        good = np.hypot(e[0], e[1]) < 0.05 and abs(d) < 0.1 and abs(e[3]) < 1e-9 and abs(e[4]) < 1e-9
        ok += good
        if good: ratios.append(len(c) * 0.5 / Tq)
        else: bad.append((x, y, k, round(float(Tq), 2), tuple(round(float(v), 3) for v in e)))
print(f"граничные старты: дошли {ok}/{tot}, время/T mean {np.mean(ratios):.3f} max {max(ratios):.3f}")
for b in bad: print("не дошёл", b)
