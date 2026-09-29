"""Схема B: rollout5 (depth=3, snap=.25) из стартов с ненулевыми v, ω (уровни поля), привязка к достижимому узлу; поле T5_n16.npy. Запуск: python3 reports/diffdrive_5d_moving.py [N]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import pi
from src.atlas_dd.five_d import make_grid, rollout5, nearest_reachable
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
al = pi / 4; n = 16
g = make_grid(n, 0.125, 64, wmax=3 * al * 0.5, almax=al, dt=0.5)
T = np.load("T5_n16.npy"); mv, mw = g['mv'], g['mw']
rng = np.random.default_rng(1); ok = 0; ratios = []; bad = []; skipped = 0
for _ in range(N):
    x, y = rng.integers(-12, 13, 2) * 0.125; k = int(rng.integers(0, 64)); th = k * 2 * pi / 64
    iv, iw = int(rng.integers(-mv, mv + 1)), int(rng.integers(-mw, mw + 1))
    st0 = (x, y, th, iv * g['dv'], iw * g['dw'])
    if T[mv + iv, mw + iw, n + int(round(x / 0.125)), n + int(round(y / 0.125)), k] >= 100:
        st0, _, _ = nearest_reachable(st0, T, g)
    Tq = T[mv + iv, mw + iw, n + int(round(st0[0] / 0.125)), n + int(round(st0[1] / 0.125)), int(round(st0[2] / (2 * pi / 64))) % 64]
    if Tq >= 100: skipped += 1; continue
    c, tr = rollout5(st0, T, g, depth=3, max_steps=60, snap=0.25); e = tr[-1]
    d = (e[2] + pi) % (2 * pi) - pi
    good = np.hypot(e[0], e[1]) < 0.05 and abs(d) < 0.1 and abs(e[3]) < 1e-9 and abs(e[4]) < 1e-9
    ok += good
    if good: ratios.append(len(c) * g['dt'] / max(Tq, 1e-9))
    else: bad.append((tuple(round(float(v), 3) for v in st0), round(float(Tq), 2), tuple(round(float(v), 3) for v in e)))
print(f"N={N}: пропущено (нет достижимого соседа) {skipped}, дошли {ok}/{N - skipped}, время/T mean {np.mean(ratios):.3f} min {min(ratios):.3f} max {max(ratios):.3f}")
for b in bad: print("не дошёл", b)
