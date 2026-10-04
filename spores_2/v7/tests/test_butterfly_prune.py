"""PRUNE=1 в butterfly_dp.solve_arcs не меняет набор сошедшихся пар (hub-v5chain-worker-16)."""
import importlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _pairs(prune):
    os.environ['PRUNE'] = str(prune)
    import src.cells7.butterfly_dp as B
    B = importlib.reload(B)
    rng = np.random.default_rng(1); A = B.Atlas(600)
    Y = np.c_[rng.uniform(-np.pi, np.pi, (1500, 2)), rng.uniform(-3, 3, (1500, 2))]
    r = A._pairs_chunk(0, Y, None, 1500)
    return None if r is None else [np.asarray(v) for v in r]


def test_prune_same_pairs():
    a, b = _pairs(0), _pairs(1)
    os.environ['PRUNE'] = '0'
    assert a is not None and b is not None
    assert len(a[0]) > 0 and abs(len(a[0]) - len(b[0])) <= max(1, int(0.02 * len(a[0])))   # допуск 2% (отсечение теряет ≤0.8% сошедших)
    key = lambda r: set(zip(r[0].tolist(), r[1].tolist()))
    assert len(key(a) & key(b)) >= 0.97 * len(key(a))
