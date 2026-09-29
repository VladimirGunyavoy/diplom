"""A2c: дуги с концом в узле — T не хуже Дейкстры без дуг, не ниже нижней границы, дуги что-то дают. Запуск: python3 tests/test_atlas_dd_arcedges.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import hypot, pi
from src.atlas_dd.lattice import solve, arc_edges
from src.atlas.solve import cost_to_go


def test_arc_edges():
    n = 3
    nodes, edges, T, pol, g = solve(n, 1.0)
    e2, ns = arc_edges(nodes, edges, 1.0)
    T2, _ = cost_to_go(e2, g)
    assert ns > 0 and T2[g] == 0.0 and all(T2[k] <= T[k] + 1e-12 for k in T)
    assert max(T[k] - T2[k] for k in T) > 0.5
    for k, t in T2.items():
        x, y, th = nodes[k]
        assert t >= max(hypot(x, y), abs((th + pi) % (2 * pi) - pi)) - 1e-9, k


if __name__ == "__main__":
    test_arc_edges(); print("OK")
