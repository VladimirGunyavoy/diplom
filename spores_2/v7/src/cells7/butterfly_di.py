"""Споры-«бабочки» на ДИ (hub-v5chain-research-8, `reports/research/butterfly_di.py`) — быстрый решатель Беллмана.
Дуги (узел → спора, точка отрезка) от V не зависят: строим пары ОДИН раз (node, k, j0, w, t), дальше Якоби векторно (reduceat по узлам)."""
import sys, time
import numpy as np
sys.path.insert(0, '../v5chain/reports/research'); sys.path.insert(0, '.')
from butterfly_di import Butterfly, BIG


class ButterflyFast(Butterfly):
    def build_pairs(s):
        nodes, ks, js, ws, ts = [], [], [], [], []
        pos = (s.sq + s.r) / (2 * s.r) * (s.m - 1); j0q = np.clip(np.floor(pos).astype(int), 0, s.m - 2); wq = pos - j0q
        for i in range(s.K):
            for j, sv in enumerate(s.sn):
                if s.ingoal[i, j]: continue
                y = s.C[i] + np.r_[0, sv]; k = np.array(s.tree.query_ball_point(y, s.tau * (abs(y[1]) + 1) + s.r + .05)); k = k[k != i]
                if not len(k): continue
                t, _ = s.arcs(y, s.C[k], s.sq); a, b = np.nonzero(np.isfinite(t))
                if not len(a): continue
                nodes.append(np.full(len(a), i * s.m + j)); ks.append(k[a]); js.append(j0q[b]); ws.append(wq[b]); ts.append(t[a, b])
        s.pn, s.pk, s.pj, s.pw, s.pt = (np.concatenate(x) for x in (nodes, ks, js, ws, ts))
        s.pidx = s.pk * s.m + s.pj; s.starts = np.r_[0, np.nonzero(np.diff(s.pn))[0] + 1]; s.pnode = s.pn[s.starts]

    def solve(s, it=3000, tol=1e-9, log=None):
        s.build_pairs(); V = s.V.ravel().copy(); G = s.ingoal.ravel()
        for n in range(it):
            val = s.pt + (1 - s.pw) * V[s.pidx] + s.pw * V[s.pidx + 1]
            mn = np.minimum.reduceat(val, s.starts); Vn = V.copy(); Vn[s.pnode] = np.minimum(V[s.pnode], mn); Vn[G] = 0.
            d = np.max(np.abs(np.minimum(Vn, BIG) - np.minimum(V, BIG))); V = Vn
            if log and n % 100 == 0: log(n, d)
            if d < tol: break
        s.V = V.reshape(s.K, s.m); s.n_it = n; s.npairs = len(s.pt); return s
