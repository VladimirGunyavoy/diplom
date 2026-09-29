"""Схема B: T из покоя по nth (16/32) и dt (0.5) при h=0.125, n=12. Запуск: python3 reports/diffdrive_5d_nth.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import pi
from src.atlas_dd.five_d import make_grid, solve5
for nth in (16, 32):
    g = make_grid(12, 0.125, nth, dt=0.5)
    T, it = solve5(g, iters=400); mv, mw, n = g['mv'], g['mw'], g['n']
    f = lambda x, y, th: T[mv, mw, n + int(round(x / 0.125)), n + int(round(y / 0.125)), int(round(th / (2 * pi / nth))) % nth]
    print(f"nth={nth} it={it}: (1,0,0)={f(1,0,0):.3f} (-1,.5,0)={f(-1,.5,0):.3f} (.5,.5,π/2)={f(.5,.5,pi/2):.3f} (-.75,-.75,π)={f(-.75,-.75,pi):.3f}", flush=True)
