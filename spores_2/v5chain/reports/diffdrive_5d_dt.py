"""Схема B: согласованность по dt при постоянных ограничениях (vmax=1, amax=1, wmax=π/2, almax=π/4): h=amax·dt²/2, nth=2π/(almax·dt²/2). T из покоя на общих стартах. Запуск: python3 reports/diffdrive_5d_dt.py <dt> <L> (L — полуразмер поля)"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import pi
from src.atlas_dd.five_d import make_grid, solve5
dt, L = float(sys.argv[1]), float(sys.argv[2]); al = pi / 4
h = dt * dt / 2; n = int(round(L / h)); nth = int(round(2 * pi / (al * dt * dt / 2)))
g = make_grid(n, h, nth, wmax=pi / 2, almax=al, dt=dt)
t0 = time.time(); T, it = solve5(g, iters=800); mv, mw = g['mv'], g['mw']
print(f"dt={dt} h={h} n={n} nth={nth} mv={mv} mw={mw} it={it} {time.time()-t0:.0f}s", flush=True)
for x, y, th in [(1, 0, 0), (0, 1, 0), (-1, 0.5, 0), (1, 1, pi / 2), (-1, -1, pi), (1.5, -1, -pi / 2)]:
    if abs(x) > L or abs(y) > L: continue
    ix, iy = n + int(round(x / h)), n + int(round(y / h)); k = int(round(th / (2 * pi / nth))) % nth
    print(f"dt={dt} ({x},{y},{th:.2f}): T={T[mv, mw, ix, iy, k]:.3f}", flush=True)
