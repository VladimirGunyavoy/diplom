"""блок 5 (3): cKDTree(boxsize) вместо 81 периодического образа в build_back. TreeBS — обёртка с интерфейсом cKDTree.query (вход — УЖЕ масштабированные точки, как у tree_query/back_heuristic).
Периодические оси: boxsize = per/scale (данные берутся по модулю); непериодические: бокс 1000 со сдвигом +500 (точки ≪ 1000)."""
import numpy as np
from scipy.spatial import cKDTree
class TreeBS:
    def __init__(self, S, P):
        sc = np.asarray(S.scale, float); self.per = np.asarray(S.per) > 0; self.box = np.where(self.per, np.asarray(S.per, float) / sc, 1000.)
        self.t = cKDTree(self._f(np.asarray(P, float) / sc), boxsize=self.box)
    def _f(self, Z):
        Z = np.array(Z, float); Z[..., self.per] = Z[..., self.per] % self.box[self.per]; Z[..., ~self.per] += 500.; return Z
    @property
    def data(self): return self.t.data
    def query(self, q, k=1, **kw): return self.t.query(self._f(q), k=k, **kw)
def build_back_bs(S, tau, NB, rho_b, sw_w=10.0):
    """как adaptive_nd.build_back, но KD-дерево — TreeBS (один экземпляр узлов, без периодических образов)"""
    from src.atlas6.adaptive_nd import _tree
    Pb, Gb, parb, layb = _tree(S, tau, S.goal_pts, NB, rho_b, sw_w, False); seqs = []
    for i in range(len(Pb)):
        seq = []; j = i
        while parb[j] >= 0: seq.append(layb[j]); j = parb[j]
        seqs.append(seq)
    return Pb, Gb, parb, layb, seqs, TreeBS(S, np.array(Pb))
