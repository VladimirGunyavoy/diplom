"""Схема B: rollout5 (depth=3, snap=0.25) на N случайных стартах из покоя на поле T5_n16.npy: доля дошедших (xy<0.05, |θ|<0.1, покой), время/T. Запуск: python3 reports/diffdrive_5d_random.py [N]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import pi
from src.atlas_dd.five_d import make_grid, rollout5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
al = pi / 4; n = 16
g = make_grid(n, 0.125, 64, wmax=3 * al * 0.5, almax=al, dt=0.5)
T = np.load("T5_n16.npy"); mv, mw = g['mv'], g['mw']
rng = np.random.default_rng(0); ok = 0; ratios = []; bad = []
for _ in range(N):
    while True:
        x, y = (rng.integers(-14, 15, 2)) * 0.125; k = int(rng.integers(0, 64)); th = k * 2 * pi / 64
        Tq = T[mv, mw, n + int(round(x / 0.125)), n + int(round(y / 0.125)), k]
        if Tq < 100: break                       # только достижимая подрешётка покоя (см. отчёт: 50% узлов покоя недостижимы по чётности)
    c, tr = rollout5((x, y, th, 0, 0), T, g, depth=3, max_steps=60, snap=0.25); e = tr[-1]
    d = (e[2] + pi) % (2 * pi) - pi
    good = np.hypot(e[0], e[1]) < 0.05 and abs(d) < 0.1 and abs(e[3]) < 1e-9 and abs(e[4]) < 1e-9
    ok += good
    if good: ratios.append(len(c) * g['dt'] / Tq)
    else: bad.append((round(float(x), 3), round(float(y), 3), round(th, 2), round(float(Tq), 2), [round(float(v), 3) for v in e]))
print(f"N={N}: дошли {ok}, время/T mean {np.mean(ratios):.3f} min {min(ratios):.3f} max {max(ratios):.3f}")
for b in bad: print("не дошёл", b)
