"""Схема B (5D) против A: T из покоя (v=ω=0) в точках (x,y,θ∈{0,π/2,π,−π/2}) на [−3,3]². Запуск: python3 reports/diffdrive_5d_vs_A.py"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas_dd.lattice import solve, arc_sweeps, HEADINGS, heading_index
from src.atlas_dd.five_d import make_grid, solve5

h, n = 0.5, 6
nodes, edges, TA, _, gA = solve(n, h)
TA, _, _ = arc_sweeps(nodes, TA, n, h, goal=gA)
pts = [(2, 0, 0), (0, 2, 0), (-2, 0, 0), (2, 2, 0), (2, 0, 4), (-2, 2, 8), (1, -1, 12), (3, 0, 4)]     # (x/h..., ) индексы θ по nth=16 → θ=k·π/8
res = {}
for dt, nth in ((0.5, 16), (0.25, 16)):
    g = make_grid(n, h, nth, dt=dt)
    t0 = time.time(); T, it = solve5(g, iters=400); el = time.time() - t0
    mv, mw = g['mv'], g['mw']
    rows = []
    for (ix, iy, kt) in pts:
        th = kt * 2 * np.pi / nth
        ka, _ = heading_index(th)
        tA = TA[(ix, iy, ka)] if (ix, iy, ka) in TA else np.nan
        tB = T[mv, mw, n + ix, n + iy, kt]
        rows.append((ix * h, iy * h, round(th, 3), round(float(tA), 3), round(float(tB), 3)))
    res[f"dt={dt}"] = dict(iters=it, sec=round(el, 1), rows=rows)
    print(f"dt={dt} iters={it} {el:.1f}s"); [print(r) for r in rows]
json.dump(res, open(os.path.join(os.path.dirname(__file__), "diffdrive_5d_vs_A.json"), "w"), indent=1)
