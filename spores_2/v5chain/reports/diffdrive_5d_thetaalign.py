"""Схема B: θ-решётка, кратная шагу поворота. almax=π/4, dt=.5 → шаг θ = almax·dt²/2 = 2π/64; wmax=3·dω. nth=64 (совпадает) против nth=48/32 (не кратно). h=.125, n=8. Запуск: python3 reports/diffdrive_5d_thetaalign.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import pi
from src.atlas_dd.five_d import make_grid, solve5
al = pi / 4; w = 3 * al * 0.5
for nth in (32, 48, 64):
    g = make_grid(8, 0.125, nth, wmax=w, almax=al, dt=0.5)
    T, it = solve5(g, iters=400); mv, mw, n = g['mv'], g['mw'], g['n']
    f = lambda x, y, th: T[mv, mw, n + int(round(x / 0.125)), n + int(round(y / 0.125)), int(round(th / (2 * pi / nth))) % nth]
    print(f"nth={nth} mw={mw} it={it}: (.75,0,0)={f(.75,0,0):.3f} (-.75,.5,0)={f(-.75,.5,0):.3f} (.5,.5,π/2)={f(.5,.5,pi/2):.3f} (-.5,-.5,π)={f(-.5,-.5,pi):.3f}", flush=True)
