"""ENV4 диагностика входа в «трубку» атласа (worker-34): для каждого из 20 тестовых стартов вдоль его OCP-траектории (только диагностика, не посев) — первое время, где V* конечна, доля времени до входа t_вх/T,
доля траектории внутри атласа после входа; расстояние (x, y, θ) от теста до ближайшего обучающего старта; покрытие поля (50k точек) и тестовых стартов по слоям.
Запуск (aida, из v7/, env как у growNd + LOAD): LOAD=<pkl layers+V> TRAIN=<env4_train/env4_ref_all.json> TESTDIR=<env4/data> OUT=<json> python reports/env4/diag_entry.py"""
import os, sys, json, pickle; sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7'))
import numpy as np, growNd as G, __main__
for n_ in dir(G):
    if n_[0].isupper() and isinstance(getattr(G, n_), type): setattr(__main__, n_, getattr(G, n_))
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
Q, ref = G.starts_ref(); tr = json.load(open(os.path.expanduser(os.environ['TRAIN']))); TQ = np.array([v['x0'] for v in tr.values() if v]); TQ[:, 1] -= 2.
def ang(a): return (a + np.pi) % (2 * np.pi) - np.pi
rows = []
for i in range(len(Q)):
    z = np.load(os.path.join(os.path.expanduser(os.environ['TESTDIR']), 'env4_q%d.npz' % i)); X = z['X'].T.copy(); T = float(z['T']); X[:, 1] -= 2.; X[:, 2] = ang(X[:, 2]); N = len(X) - 1
    v = A.vstar(X); fin = v < G.BIG / 2; k0 = int(np.argmax(fin)) if fin.any() else None
    dq = np.sqrt((TQ[:, 0] - Q[i, 0]) ** 2 + (TQ[:, 1] - Q[i, 1]) ** 2 + ang(TQ[:, 2] - Q[i, 2]) ** 2)
    rows.append(dict(i=i, T=T, t_in=None if k0 is None else T * k0 / N, frac_in=None if k0 is None else k0 / N, frac_inside_after=None if k0 is None else float(fin[k0:].mean()), v0_finite=bool(fin[0]),
                     V0=float(v[0]) if fin[0] else None, dist_train_xyth=float(dq.min())))
    print(rows[-1], flush=True)
rng = np.random.default_rng(5); P = np.array([G.rand_seed(rng) for _ in range(50000)]); P = P[[G.inbox(p) and G.free(p) for p in P]]; cov = []
for l in A.layers:
    idx = G.HexIdx()
    for c in l: idx.add(c)
    cov.append((float(idx.covered(G.wrapy(P)).mean()), int(idx.covered(G.wrapy(Q)).sum())))
out = dict(rows=rows, cov_field_and_tests_per_layer=cov, vfinite_nodes=float((A.V < G.BIG / 2).mean())); json.dump(out, open(os.path.expanduser(os.environ['OUT']), 'w'))
fi = [r['frac_in'] for r in rows if r['frac_in'] is not None]; print('вход найден у', len(fi), 'из', len(rows), '| t_вх/T мед', np.median(fi) if fi else None, '| покрытие поля по слоям', [round(c[0], 4) for c in cov], '| тестов покрыто', [c[1] for c in cov])
