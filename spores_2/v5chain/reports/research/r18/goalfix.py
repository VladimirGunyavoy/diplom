# research-18: в 4D узлы почти не попадают в коробку цели ±.05 (66 конечных V из 437k) — цель атласа = коробка ±GOALR, V₀ = время финиша стрельбой (finish_gen.shoot) до настоящей цели.
import os, sys, pickle, time, numpy as np
import bfs_gen as B
G = B.G; import finish_gen as FG
R = float(os.environ.get('GOALR', .35)); d = pickle.load(open(os.environ['LOAD'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish()
box = np.flatnonzero((np.abs(G.wrapy(A.P)) <= R).all(1)); t0 = time.time()
def _sh(y): Ts, _ = FG.shoot(y, G.f, G.US, G.RHOV, G.wrapy, float(os.environ.get('FT', 4.))); return Ts if np.isfinite(Ts) else G.BIG
from multiprocessing import Pool
with Pool(int(os.environ.get('NPOOL', 24))) as pool: V0 = np.array(pool.map(_sh, list(A.P[box]), chunksize=8))
print('коробка', R, 'узлов', len(box), 'финиш найден', int((V0 < G.BIG / 2).sum()), 'T₀ мед', round(float(np.median(V0[V0 < G.BIG / 2])), 3), round(time.time() - t0), 'с', flush=True)
A.V[box] = np.minimum(A.V[box], V0); A.solve(); f = A.V < G.BIG / 2
rq = np.random.default_rng(5); Qc = rq.uniform(-2, 2, (4000, G.N)); vq = A.vstar(G.wrapy(Qc))
Q, ref = G.starts_ref(); T, sw, _ = A.rollout(Q); fz = np.isfinite(T); print('АГЕНТ дошли', int(fz.sum()), '/ 60', 'T/T* mean', round(float((T[fz] / ref[fz]).mean()), 4) if fz.any() else None, 'med', round(float(np.median(T[fz] / ref[fz])), 4) if fz.any() else None, 'max', round(float((T[fz] / ref[fz]).max()), 3) if fz.any() else None, flush=True)
print('конечных узлов', round(float(f.mean()), 3), 'проб [-2,2]^4 накрыто', round(float((vq < G.BIG / 2).mean()), 3), 'V̂ накрытых кв.', np.quantile(vq[vq < G.BIG / 2], [.1, .5, .9]).round(2).tolist() if (vq < G.BIG / 2).any() else None, flush=True)
pickle.dump(dict(layers=A.layers, V=A.V), open(os.environ['OUTP'], 'wb'))
