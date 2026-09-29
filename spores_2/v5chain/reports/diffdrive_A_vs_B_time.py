"""Итоговое сравнение A и B по времени rollout на одних стартах (из покоя для B). A: rollout_multi (h=.5 с дугами + тонкое h=.125), B: rollout5_best на поле n=16 (времена из diffdrive_5d_snap: 2.0/5.0/6.5/6.0/8.5/8.0). Запуск: python3 reports/diffdrive_A_vs_B_time.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import pi, hypot
from src.atlas_dd.lattice import solve, arc_sweeps
from src.atlas_dd.plan import rollout_multi
n1, h1 = 8, 0.5
nodes, edges, T1, _, g = solve(n1, h1); T1, _, _ = arc_sweeps(nodes, T1, n1, h1, goal=g)
n2, h2 = 8, 0.125
nodes2, edges2, T2, _, g2 = solve(n2, h2); T2, _, _ = arc_sweeps(nodes2, T2, n2, h2, goal=g2)
B = {(0.75, 0, 0): 2.0, (-0.75, 0.5, 0): 5.0, (1.5, -1.0, pi / 2): 6.5, (-1.5, 1.0, pi): 6.0, (2.0, 2.0, -pi / 2): 8.5, (-1.0, -1.5, 0.0): 8.0}
for st, tb in B.items():
    c, tr = rollout_multi(st, [(T1, n1, h1), (T2, n2, h2)]); e = tr[-1]
    ta = sum(t for _, t in c)
    print(f"{tuple(round(v, 2) for v in st)}: A время {ta:.2f} (конец xy {hypot(e[0], e[1]):.3f}), B время {tb:.1f}, B/A {tb / ta:.2f}", flush=True)
