"""research-37 (graph_to_cells, замер senior-2 (в)): связные компоненты kNN-графа с ОДИНАКОВЫМ argmin-управлением.
У каждого узла u*(i) = базовое управление (без τ) из argmin_a [τ_a + Σλ W(flow(i, a, τ_a))] (как в VI-LP; ребро в цель — GL).
Компоненты: узлы i, j соединены, если j ∈ kNN(i) (k = 2d+4) и u*(i) = u*(j). Числа: компонент на 95% активных узлов, размеры,
доля узлов на границе (есть сосед с другим u*). Смысл: внутри компоненты W = τ + W(flow) точно ⇒ хранить только выходное сечение.
Запуск: как graph_cells.py (SRC=npz с P,W или строит случайные)."""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import tvg_vt as T
env, d = T.env, T.d
VCAP = float(os.environ.get('VCAP', 40)); KG = int(os.environ.get('KG', 2 * d + 4))

def main():
    ts = T.TS; S = T.mkS(ts); z = np.load(os.path.expanduser(os.environ['SRC'])); P, W = z['P'], np.minimum(z['W'], VCAP); n = len(P)
    from tvg_lp import InterpLP
    tree = T.cKDTree(T._scaled(S, T._images(S, P))); ip = InterpLP(S, tree, n)
    E, OK, GL = T.edges(P, S); L2 = OK.shape[1]; J, lam, ins = ip.weights(E.reshape(-1, d)); sub = (lam * W[J]).sum(1).reshape(-1, L2)
    cand = np.where(GL > 0, GL, T.TAUV[None] + sub); cand[~OK | ((GL <= 0) & ~ins.reshape(-1, L2))] = 1e9
    a = np.argmin(cand, 1); u = T.SIDX[a]; act = (W < VCAP - 1) & ~T.in_goal(P) & (cand.min(1) < 1e8)
    _, nb = tree.query(S.wrap(P) / S.scale, k=KG + 1); nb = nb[:, 1:] % n
    I = np.repeat(np.arange(n), KG); Jn = nb.ravel(); m = act[I] & act[Jn] & (u[I] == u[Jn])
    G = coo_matrix((np.ones(m.sum()), (I[m], Jn[m])), shape=(n, n)); nc, lab = connected_components(G, directed=False)
    lab[~act] = -1; sz = np.bincount(lab[act]); szs = np.sort(sz)[::-1]; cum = np.cumsum(szs) / act.sum(); k95 = int(np.searchsorted(cum, .95) + 1)
    bnd = act & ((u[nb] != u[:, None]) & act[nb]).any(1)
    res = dict(env=T.ENV, nodes=n, active=int(act.sum()), controls=int(len(np.unique(u[act]))), comps=int((sz > 0).sum()), k95=k95, top10=[int(v) for v in szs[:10]],
               singletons=int((sz == 1).sum()), frac_boundary=float(bnd.sum() / act.sum()), frac_top1=float(szs[0] / act.sum()))
    print('ИТОГ %s' % res, flush=True)
    json.dump(res, open(os.path.expanduser('~/spore_v5/r37/gctrl_%s_%d_%d.json' % (T.ENV, n, ts)), 'w'))
    np.save(os.path.expanduser('~/spore_v5/r37/gctrl_lab_%s_%d_%d.npy' % (T.ENV, n, ts)), lab)

if __name__ == '__main__': main()
