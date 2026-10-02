"""research hub-research-7: воспроизвести расхождение worker-12 (spore_v на сжатых клетках cover_layer: V*/T* 1.6–1.8 при 7×9). Мой прототип с мелкими
клетками (τ .05, r .03) против крупных (τ .4, r .1) в той же коробке; диагноз — доля узлов, у которых в интерполяции точки прихода есть BIG-угол."""
import sys, os, json, time, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_spore_di import Cover, BIG
from v7_faces_di import tstar_box
rng = np.random.default_rng(3); Q = rng.uniform(-.8, .8, (200, 2)); Ts = tstar_box(Q[:, 0], Q[:, 1], .1); ok = Ts > .05
for tau, r, seeds in [tuple(map(float, a.split(','))) for a in sys.argv[1:]]:
    t0 = time.time(); Cv = Cover(L=1.2, tau=tau, r=r, seeds=int(seeds)); V = Cv.solve(); Vs = Cv.Vstar(Q)
    P = np.stack([Cv.X.ravel(), Cv.Vv.ravel()], 1); dt = np.repeat(Cv.C[:, 4], Cv.m * Cv.nt) * (Cv.tgn[1] - Cv.tgn[0]); bigc = []
    for u in (1., -1.):
        qi, k, idx, w = Cv.pairs(Cv.phi(P, u, dt)); hb = ((V[idx] > BIG / 2) & (w > 1e-9)).any(1); bigc.append(np.bincount(qi[hb], minlength=Cv.N) > 0)
    fin = V < BIG / 2; contam = (bigc[0] | bigc[1]) & fin
    T, sw = Cv.rollout(Q); okr = ok & np.isfinite(T); rv = Vs[ok] / Ts[ok]
    print(json.dumps(dict(tau=tau, r=r, cells=int(Cv.K), nodes=int(Cv.N), iters=Cv.iters, V_start_over_Tstar=dict(med=round(float(np.nanmedian(rv)), 3), mean=round(float(np.nanmean(rv[np.isfinite(rv)])), 3)),
          share_finite_nodes_with_BIG_corner=round(float(contam.sum() / max(fin.sum(), 1)), 3), reach=round(float(np.isfinite(T).mean()), 3), T_over_Tstar_mean=round(float(np.mean(T[okr] / Ts[okr])), 3) if okr.any() else None,
          sw_med=float(np.median(sw[okr])) if okr.any() else None, sec=round(time.time() - t0))), flush=True)
