"""Г4.2: общие цели/старты/эталон. 10 целей-покоев g∈U(-1.5,1.5)² (v=0), по 50 стартов U(-2,2)^4 на цель, keep как DI4 (остановочный путь в поле 2.5);
эталон точный сдвигом: T*=max(tbox(x1-g1,v1), tbox(x2-g2,v2)) (di4_ref_exact, коробка ±.35 вокруг цели)."""
import numpy as np
from di4_ref_exact import tbox
def make(NG=10, NS=50, seed=42):
    rg = np.random.default_rng(seed); G = rg.uniform(-1.5, 1.5, (NG, 2)); out = []
    for gi in range(NG):
        Q = np.random.default_rng(100 + gi).uniform(-2, 2, (NS, 4)); g = G[gi]
        ref = np.array([max(tbox(q[0] - g[0], q[2]), tbox(q[1] - g[1], q[3])) for q in Q])
        keep = np.abs(np.c_[Q[:, 0] + Q[:, 2] * abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * abs(Q[:, 3]) / 2]).max(1) <= 2.5
        out.append((np.r_[g, 0., 0.], Q, ref, keep))
    return out
