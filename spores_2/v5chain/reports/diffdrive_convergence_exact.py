"""Сходимость по h для дуг с концом в узле (arc_edges): T на общих узлах, поле [−3,3]². Запуск: python3 reports/diffdrive_convergence_exact.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import hypot, pi
from src.atlas_dd.lattice import solve, arc_edges, HEADINGS
from src.atlas.solve import cost_to_go

Ts, N = {}, {}
for h in (1.0, 0.5, 0.25):
    n = int(3 / h)
    nodes, edges, T0, _, g = solve(n, h)
    e2, _ = arc_edges(nodes, edges, h)
    T, _ = cost_to_go(e2, g)
    Ts[h] = (T, T0, n)
    for k, t in T.items():
        x, y, th = nodes[k]
        lb = max(hypot(x, y), abs((th + pi) % (2 * pi) - pi))
        assert t >= lb - 1e-9, (h, k, t, lb)
for name, ix in (("без дуг", 1), ("дуги в узле", 0)):
    for a, b in ((1.0, 0.5), (0.5, 0.25)):
        Ta, Tb = Ts[a][ix], Ts[b][ix]
        na, nb = Ts[a][2], Ts[b][2]
        f = int(round(a / b))
        d = [Ta[(i, j, k)] - Tb[(i * f, j * f, k)] for (i, j, k) in Ta]
        print(f"{name} {a}->{b}: mean {sum(d)/len(d):.4f} max {max(d):.3f} min {min(d):.3f}")
