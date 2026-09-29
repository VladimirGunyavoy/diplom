"""Узлы и рёбра решётки спор (АТЛАС §3–4)."""
import numpy as np
from collections import defaultdict
from .coords import A


def key(i, j, s, C, D):
    return (i, j, 0) if np.isclose(C[i], D[j]) else (i, j, s)


def build_lattice(C, D, a=A, vmax=None):
    nodes = {}
    for i, c in enumerate(C):
        for j, d in enumerate(D):
            if d < c and not np.isclose(d, c):
                continue
            for s in (+1, -1):
                k = key(i, j, s, C, D)
                if k in nodes:
                    continue
                v = s * np.sqrt(max(a * (d - c), 0.0))
                if vmax is not None and abs(v) > vmax:
                    continue
                nodes[k] = ((c + d) / 2, v)
    return nodes


def build_edges(nodes, a=A):
    plus, minus = defaultdict(list), defaultdict(list)
    for k, (x, v) in nodes.items():
        i, j, _ = k
        plus[i].append((v, k))
        minus[j].append((v, k))
    edges = defaultdict(list)
    for lst in plus.values():
        lst.sort()
        for (v1, k1), (v2, k2) in zip(lst, lst[1:]):
            edges[k1].append((k2, (v2 - v1) / a, +1))
    for lst in minus.values():
        lst.sort(reverse=True)
        for (v1, k1), (v2, k2) in zip(lst, lst[1:]):
            edges[k1].append((k2, (v1 - v2) / a, -1))
    return edges
