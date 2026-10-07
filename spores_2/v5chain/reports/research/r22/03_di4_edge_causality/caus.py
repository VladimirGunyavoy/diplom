"""research-22, эксп. 03: почему вёдра каскадируют в 4D. На сошедшемся V (SBCAUS, SBLAY 0, di4 L400) для каждого ребра: причинно ли оно
(все вершины с весом > 1e-6 имеют V < V_узла), на сколько вершины выше узла, отдельно — для лучшего ребра узла (на котором достигается V)."""
import os, pickle, numpy as np
from tqdm import tqdm
E = os.environ['SAVEE']; I = np.load(E + '_I.npy'); IDX = np.load(E + '_IDX.npy', mmap_mode='r'); W = np.load(E + '_W.npy', mmap_mode='r'); V = pickle.load(open('V.pkl', 'rb')); BIG = float(os.environ.get("BIGV", 1e3)); DTN = .1; PESS = 1.
n = len(I); val = np.empty(n); up = np.empty(n); nunk = np.empty(n, np.int8); ex = np.empty(n, bool); wup = np.empty(n)
for a in tqdm(range(0, n, 1 << 20), desc='edges', mininterval=10):
    b = min(n, a + (1 << 20)); w = np.asarray(W[a:b], float); vi = V[np.asarray(IDX[a:b])]; ok = vi < BIG / 2; act = w > 1e-6; wk = w * ok; sm = wk.sum(1)
    with np.errstate(invalid='ignore'): v = (wk * np.where(ok, vi, 0.)).sum(1) + (1 - sm) * (np.where(ok, vi, -np.inf).max(1) + PESS)
    v[sm < .5] = BIG; val[a:b] = DTN + v; vn = V[I[a:b]][:, None]
    up[a:b] = np.where(act & ok, vi - vn, -np.inf).max(1); nunk[a:b] = (act & ~ok).sum(1); ex[a:b] = (w[:, 0] == 1.) & (w[:, 1:] == 0).all(1); wup[a:b] = (w * (act & ok & (vi >= vn))).sum(1)
fin = (V[I] < BIG / 2) & (val < BIG / 2); best = fin & (val <= V[I] + 1e-9)
def rep(name, m):
    u = up[m]; print('%-34s рёбер %8d | непричинных (вершина ≥ узла) %.3f | вес таких вершин mean %.3f | max(V_верш) − V_узла: мед. %+.3f p90 %+.3f p99 %+.3f | с неизвестной вершиной (PESS) %.3f' % (name, m.sum(), (u >= 0).mean(), wup[m].mean(), np.median(u), *np.percentile(u, [90, 99]), (nunk[m] > 0).mean()), flush=True)
rep('все конечные', fin); rep('точные по столбцу', fin & ex); rep('стенсилы', fin & ~ex); rep('ЛУЧШИЕ (дают V узла)', best); rep('лучшие: точные', best & ex); rep('лучшие: стенсилы', best & ~ex)
nb = np.unique(I[best]); print('узлов с лучшим ребром', len(nb), '| доля узлов, у которых лучшее ребро точное: %.3f' % (len(np.unique(I[best & ex])) / len(nb)), '| лучшее причинное: %.3f' % (len(np.unique(I[best & (up < 0)])) / len(nb)))
