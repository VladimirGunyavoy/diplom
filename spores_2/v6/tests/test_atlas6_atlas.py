"""Атлас DI v6: покрытие, соседи, дыры на выходах. Запуск: python3 tests/test_atlas6_atlas.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.atlas import Atlas, seed_lattice

def test_coverage_and_neighbors():
    a = Atlas(seed_lattice((-2, 2), (-2, 2), 0.5, 0.5), r=0.4, tau=0.6, alpha=0.7)
    cov = a.coverage((-1.5, 1.5), (-1.5, 1.5)); cor = a.coverage((-1.5, 1.5), (-1.5, 1.5), core=True)
    print(f"клеток {len(a.cells)}, покрытие {cov:.3f}, ядра {cor:.3f}, дыр на выходах {len(a.exit_gaps())}")
    assert cov > 0.99
    nb = a.neighbors()
    assert all(len(s) > 0 for s in nb) and np.mean([len(s) for s in nb]) > 3

def test_sparse_has_gaps():
    a = Atlas(seed_lattice((-2, 2), (-2, 2), 2.0, 2.0), r=0.1, tau=0.2, alpha=0.7)
    assert a.coverage((-2, 2), (-2, 2)) < 0.5 and len(a.exit_gaps()) > 0        # редкие споры — дыры, туда сеять

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
