"""A5-lite: двухуровневое поле (h=1 и h=0.25 у цели) даёт финиш < 0.1 по (x,y). Запуск: python3 tests/test_atlas_dd_multi.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from math import hypot, pi
from src.atlas_dd.lattice import solve, arc_sweeps
from src.atlas_dd.plan import rollout_multi


def test_multi():
    n, nf, hf = 4, 4, 0.25
    nd, ed, T, p, g = solve(n, 1.0); T, _, _ = arc_sweeps(nd, T, n, goal=g)
    nd2, ed2, Tf, p2, g2 = solve(nf, hf); Tf, _, _ = arc_sweeps(nd2, Tf, nf, h=hf, goal=g2)
    for st in [(2.5, -1.5, 1.0), (-2.5, 1.8, -2.0)]:
        ctrl, tr = rollout_multi(st, [(T, n, 1.0), (Tf, nf, hf)], refine_dt=True, lookahead=3, la_dt=0.1)
        e = tr[-1]
        assert hypot(e[0], e[1]) < 0.1 and abs((e[2] + pi) % (2 * pi) - pi) < 0.1, (st, e)


if __name__ == "__main__":
    test_multi(); print("OK")
