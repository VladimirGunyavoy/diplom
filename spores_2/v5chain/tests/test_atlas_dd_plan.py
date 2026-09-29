"""A4: rollout приводит в окрестность цели (0.4) из нескольких стартов. Запуск: python3 tests/test_atlas_dd_plan.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import hypot, pi
from src.atlas_dd.lattice import solve, arc_sweeps
from src.atlas_dd.plan import rollout


def test_plan():
    n = 4
    nodes, edges, T, pol, g = solve(n, 1.0)
    T, _, _ = arc_sweeps(nodes, T, n, goal=g)
    for st in [(2.5, -1.5, 1.0), (-3.0, 2.0, -2.0), (1.2, 3.1, 0.3)]:
        ctrl, tr = rollout(st, T, n)
        e = tr[-1]
        assert hypot(e[0], e[1]) < 0.4 and abs((e[2] + pi) % (2 * pi) - pi) < 0.4, (st, e)
        assert ctrl


if __name__ == "__main__":
    test_plan(); print("OK")
