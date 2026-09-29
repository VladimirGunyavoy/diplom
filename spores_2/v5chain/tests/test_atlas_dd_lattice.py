"""A2: решётка дифдрайва — ручные значения и согласованность Беллмана. Запуск: python3 tests/test_atlas_dd_lattice.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas_dd.lattice import solve, VECS, HEADINGS, NH


def test_lattice():
    nodes, edges, T, pol, g = solve(6)
    assert len(T) == len(nodes)                                   # все узлы достижимы
    k0 = g[2]; assert HEADINGS[k0] == 0.0
    assert abs(T[(3, 0, k0)] - 3.0) < 1e-12                       # назад по курсу 0
    k21 = VECS.index((2, 1))
    assert abs(T[(2, 1, k21)] - 5 ** 0.5 - abs(HEADINGS[k21]) ) < 1e-12   # назад √5 и поворот к курсу 0 (через промежуточные курсы = |Δθ|)
    for u, lst in edges.items():                                  # Беллман: T[u] ≤ cost + T[v]
        for v, c, _ in lst:
            assert T[u] <= c + T[v] + 1e-12


if __name__ == "__main__":
    test_lattice(); print("OK")
