"""Адаптация для дифдрайва: где интерполяция грубого поля (h=0.5, дуги в узле A2c) врёт против тонкого (h=0.25)? Ошибка на узлах тонкого, не лежащих в грубом, по кольцам расстояния до цели. Запуск: python3 reports/diffdrive_adapt_indicator.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import hypot
from src.atlas_dd.lattice import solve, arc_edges, interp_T3
from src.atlas.solve import cost_to_go

def field(h, n):
    nodes, edges, _, _, g = solve(n, h); e2, _ = arc_edges(nodes, edges, h); T, _ = cost_to_go(e2, g); return nodes, T
nc, Tc = field(0.5, 4); nf, Tf = field(0.25, 8)
rings = {}
for (i, j, k), (x, y, th) in nf.items():
    if i % 2 == 0 and j % 2 == 0: continue                    # узлы, совпадающие с грубыми, пропускаем
    if abs(x) > 1.9 or abs(y) > 1.9: continue
    e = interp_T3(x, y, th, Tc, 4, 0.5) - Tf[(i, j, k)]
    r = min(int(hypot(x, y) / 0.5), 3); rings.setdefault(r, []).append(e)
for r in sorted(rings):
    a = np.array(rings[r]); print(f"r∈[{0.5*r:.1f},{0.5*r+0.5:.1f}) n={len(a)} mean {a.mean():+.3f} mean|e| {abs(a).mean():.3f} max|e| {abs(a).max():.3f}")
