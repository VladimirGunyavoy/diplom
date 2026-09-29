"""Схема B (5D): цель 0, T не растёт, нижняя граница по позиции (d ≥ ... время не меньше, чем при |v|≤vmax). Запуск: python3 tests/test_atlas_dd_5d.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas_dd.five_d import make_grid, solve5


def test_5d():
    g = make_grid(4, 0.5, 8)
    T, it = solve5(g, iters=60)
    mv, mw, n = g['mv'], g['mw'], g['n']
    assert T[mv, mw, n, n, 0] == 0.0 and T.min() >= 0
    # из покоя в (1,0,θ=0): не быстрее d/vmax
    t = T[mv, mw, n + 2, n, 0]
    assert 1.0 <= t < 100, t


if __name__ == "__main__":
    test_5d(); print("OK")
