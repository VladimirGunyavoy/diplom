# research-18: в 4D V не растекается — interp ставит BIG, если ЛЮБАЯ из 2^N вершин BIG (16 в 4D). Проба: веса по конечным вершинам, BIG только при сумме весов < WTHR.
import os, sys, pickle, time, numpy as np
import bfs_gen as B
G = B.G; WT = float(os.environ.get('WTHR', .5))
PD = float(os.environ.get('PESS', -1))
def interp_ren(W, VI):
    ok = VI < G.BIG / 2; w = W * ok; s = w.sum(1)
    if PD >= 0:                                                                       # пессимистично: неизвестная вершина = max известных + PD (Якоби только уменьшает V ⇒ верхняя оценка уточнится)
        mx = np.where(ok, VI, -np.inf).max(1); v = (w * np.where(ok, VI, 0.)).sum(1) + (1 - s) * (mx + PD)
    else: v = (w * np.where(ok, VI, 0.)).sum(1) / np.maximum(s, 1e-12)
    v[s < WT] = G.BIG; return v
