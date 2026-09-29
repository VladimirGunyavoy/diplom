"""Эталон АТЛАС §6: сетка 81x81 на [-4, 4], a=1, цель — начало координат. Запуск: python3 tests/test_atlas_core.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas import key, build_lattice, build_edges, cost_to_go, interp_T, T_star


def run():
    C = np.linspace(-4, 4, 81)
    D = C.copy()
    nodes = build_lattice(C, D)
    edges = build_edges(nodes)
    i0 = int(np.argmin(abs(C)))
    T, _ = cost_to_go(edges, key(i0, i0, 0, C, D))
    err = max(abs(T[k] - T_star(*nodes[k])) for k in T)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(2000):
        x, v = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        ti = interp_T(x, v, C, D, T)
        if not np.isnan(ti):
            errs.append(abs(ti - T_star(x, v)))
    return len(nodes), len(T), err, np.array(errs)


def test_atlas_core():
    n, r, err, errs = run()
    print("nodes", n, "reached", r, "max node err", err)
    print("interp: n", len(errs), "mean", errs.mean(), "max", errs.max())
    assert n == 6561 and r == 6561
    assert err <= 1e-6
    assert abs(errs.mean() - 0.0099) <= 0.0099 * 0.1
    assert abs(errs.max() - 0.236) <= 0.236 * 0.1


if __name__ == "__main__":
    test_atlas_core()
    print("OK")
