"""A2b: дуги — T не растёт, в цели 0, не ниже нижней границы max(d/vmax, |θ|/ωmax). Запуск: python3 tests/test_atlas_dd_arcs.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math import hypot, pi
from src.atlas_dd.lattice import solve, arc_sweeps, NH


def test_arcs():
    n = 3
    nodes, edges, T, pol, g = solve(n, 1.0)
    T2, it, hist = arc_sweeps(nodes, T, n, goal=g)
    assert T2[g] == 0.0 and all(T2[k] <= T[k] + 1e-12 for k in T)
    assert max(T[k] - T2[k] for k in T) > 0.1                    # дуги реально что-то дают
    for k, t in T2.items():
        x, y, th = nodes[k]
        lb = max(hypot(x, y), abs((th + pi) % (2 * pi) - pi))
        assert t >= lb - 1e-9, (k, t, lb)


if __name__ == "__main__":
    test_arcs(); print("OK")
