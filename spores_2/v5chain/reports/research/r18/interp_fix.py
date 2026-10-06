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
d = pickle.load(open(os.environ['LOAD'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = np.minimum(A.V, d['V'])
f0 = (A.V < G.BIG / 2).sum(); G.Atlas.interp = staticmethod(interp_ren); t0 = time.time(); A.solve(); f = A.V < G.BIG / 2
rq = np.random.default_rng(5); Qc = rq.uniform(-2, 2, (4000, G.N)); vq = A.vstar(G.wrapy(Qc)); cov = vq < G.BIG / 2
print('WTHR', WT, 'PESS', PD, '| конечных узлов было', int(f0), 'стало', round(float(f.mean()), 3), '| проб [-2,2]^4 накрыто', round(float(cov.mean()), 3), '| V̂ кв.', np.quantile(vq[cov], [.1, .5, .9]).round(2).tolist() if cov.any() else None, '| итераций', A.n_it, round(time.time() - t0), 'с', flush=True)
Q, ref = G.starts_ref(); vs = A.vstar(G.wrapy(Q)); ok = vs < G.BIG / 2; print('эталонных стартов накрыто', int(ok.sum()), '/ 60', '| V̂/T* мед', round(float(np.median(vs[ok] / ref[ok])), 3) if ok.any() else None, flush=True)
pickle.dump(dict(layers=A.layers, V=A.V), open(os.environ['OUTP'], 'wb'))
