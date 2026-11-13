"""блок 5 (3): эквивалентность k-NN (образы vs boxsize) на m8 NB2000 и 1000 случайных запросах + вес/время на микробенче 60000 узлов."""
import sys, os, runpy, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import bs
sys.argv = ['spores_8d.py', 'm8', '2000', '200', '3', 'eq']
G = runpy.run_path(os.path.expanduser('~/spore_v5/a31/spores_2/bench/spores_8d.py'), run_name='lib'); S, back, common = G['S'], G['back'], G['common']
from src.atlas6.adaptive_nd import tree_query, _images, _scaled
from scipy.spatial import cKDTree
Pb, Gb, parb, layb, seqs, tree = back; n = len(Pb); T2 = bs.TreeBS(S, np.array(Pb)); rng = np.random.default_rng(5); sc = np.asarray(S.scale); per = np.asarray(S.per) > 0
Q = np.where(per, rng.uniform(-np.pi, np.pi, (1000, 8)), rng.uniform(-3, 3, (1000, 8))); Q[:, ~per] = rng.uniform(-3, 3, (1000, int((~per).sum())))
def pdist(q, j):
    d = (np.asarray(Pb)[j] - q) / sc; d[:, per] = (d[:, per] + np.pi / sc[per]) % (2 * np.pi / sc[per]) - np.pi / sc[per]; return np.linalg.norm(d, axis=1)
same_j = 0; dmax = 0.; difset = 0
for q in Q:
    d1, j1 = tree_query(S, tree, Pb, q, 8); d2, j2 = tree_query(S, T2, Pb, q, 8)
    same_j += int(np.array_equal(j1, j2)); difset += int(set(j1.tolist()) != set(j2.tolist())); dmax = max(dmax, float(np.abs(pdist(q, j1) - pdist(q, j2)).max()))
print('ЭКВИВАЛЕНТНОСТЬ (m8, NB=%d узлов, 1000 запросов, kn=8): одинаковый упорядоченный набор j: %d/1000, одинаковое множество j: %d/1000, max |Δ расстояния| %.2e' % (n, same_j, 1000 - difset, dmax))
# k=1 и kn*9 прямой запрос
d1, j1 = tree.query(Q / sc, k=1); d2, j2 = T2.query(Q / sc, k=1); print('  k=1: |Δd| max %.2e, одинаковый j %d/1000 (j1%%n)' % (np.abs(d1 - d2).max(), int((j1 % n == j2).sum())))
# микробенч 60000 узлов
N = 60000; P = np.where(per, rng.uniform(-np.pi, np.pi, (N, 8)), rng.uniform(-3, 3, (N, 8))); 
t0 = time.perf_counter(); Z = _images(S, P); tr1 = cKDTree(_scaled(S, Z)); t_img = time.perf_counter() - t0
t0 = time.perf_counter(); tr2 = bs.TreeBS(S, P); t_bs = time.perf_counter() - t0
w1 = common.deep_weight(tr1); w2 = common.deep_weight(tr2)
Qb = np.where(per, rng.uniform(-np.pi, np.pi, (1000, 8)), rng.uniform(-3, 3, (1000, 8))) / sc
t0 = time.perf_counter(); [tr1.query(q, k=72) for q in Qb]; q1 = time.perf_counter() - t0; t0 = time.perf_counter(); [tr2.query(q, k=72) for q in Qb]; q2 = time.perf_counter() - t0
print('МИКРОБЕНЧ (N=60000, 8D, 4 периодических): образы — построение KD %.2f с, вес %s (%.1f МБ), 1000 запросов k=72: %.2f с; boxsize — построение %.2f с, вес %s (%.1f МБ), запросы %.2f с' % (t_img, w1, w1['bytes'] / 1e6, q1, t_bs, w2, w2['bytes'] / 1e6, q2))
