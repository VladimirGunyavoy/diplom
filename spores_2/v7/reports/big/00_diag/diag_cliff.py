"""BIG0 п.2: обрыв между 25 и 8 клетками (маятник). K самых крупных клеток → покрытие поля, V*<BIG, дошли; картинка."""
import sys, os, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; A = B.make_atlas(); vol = np.array([B.cell_vol(c) for c in A.cells]); order = np.argsort(-vol)
gx, gy = np.meshgrid(np.linspace(-np.pi, np.pi, 160), np.linspace(-3.5, 3.5, 160)); Y = np.c_[gx.ravel(), gy.ravel()]
OUT = os.environ['BIGOUT']; res = {}; maps = {}
for K in [int(k) for k in os.environ.get('KS', '25,20,16,12,10,8').split(',')]:
    A2 = B.sub(A, order[:K]); ts = B.solve(A2); s = B.evalA(A2)
    I, IDX, W = A2.stencils(Y); cov = np.zeros(len(Y), bool); cov[I] = True; V = A2.vstar(Y); reach = V < G.BIG / 2
    res[K] = dict(cells=len(A2.cells), nodes=int(A2.N), t_solve=ts, reached=s['reached'], ratio_med=s['ratio_med'], covered=float(cov.mean()), Vfinite=float(reach.mean()), Vfinite_of_covered=float(reach[cov].mean()), T=s['T'])
    maps[K] = (cov.reshape(gx.shape), reach.reshape(gx.shape)); print(K, {k: v for k, v in res[K].items() if k != 'T'}, flush=True)
    json.dump(res, open(OUT + '/pend_cliff.json', 'w'), indent=1, default=float)
np.savez(OUT + '/pend_cliff_maps.npz', gx=gx, gy=gy, **{'cov%d' % k: m[0] for k, m in maps.items()}, **{'V%d' % k: m[1] for k, m in maps.items()}, Q=B.Q)
