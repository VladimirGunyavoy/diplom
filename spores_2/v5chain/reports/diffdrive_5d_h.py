"""Схема B: зависимость T(1,0,θ=0 из покоя) от h и nth (идеал двойного интегратора: 2.0). Запуск: python3 reports/diffdrive_5d_h.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.atlas_dd.five_d import make_grid, solve5
for h, n, nth in ((0.5, 4, 16), (0.25, 8, 16), (0.25, 8, 32), (0.125, 16, 16)):
    g = make_grid(n, h, nth, dt=0.5)
    T, it = solve5(g, iters=400)
    mv, mw = g['mv'], g['mw']; i = int(round(1 / h))
    print(f"h={h} nth={nth} iters={it}: T(1,0,0)={T[mv, mw, n + i, n, 0]:.3f} T(0,1,0)={T[mv, mw, n, n + i, 0]:.3f}", flush=True)
