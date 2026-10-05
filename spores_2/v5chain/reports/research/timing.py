"""research-14: время этапов атласа (постройка / V / агент 100 стартов / сетка V 161×141) и проверка, что FASTLOC даёт те же числа."""
import numpy as np, os, time
import grow_cells2d as G
t = time.time(); A = G.Atlas(seed=0); tb = time.time() - t; out = {}
for fl in (1, 0):
    G.FASTLOC = fl; A.V = np.full(A.N, G.BIG); A.V[A.goal] = 0.
    for c in A.cells: c.__dict__.pop('bk', None)
    t = time.time(); A.solve(); ts = time.time() - t; Q, ref = G.starts_ref(); ok = ref > .05
    t = time.time(); T, sw, _ = A.rollout(Q[ok]); tr = time.time() - t
    gx, gw = np.linspace(-G.XL, G.XL, 161), np.linspace(-G.WL, G.WL, 141); GX, GW = np.meshgrid(gx, gw)
    t = time.time(); VV = A.vstar(np.c_[GX.ravel(), GW.ravel()]); tv = time.time() - t
    out[fl] = (T, VV); print('FASTLOC', fl, 'клеток', len(A.cells), 'постройка %.0f с, V %.1f с, агент %.1f с, сетка V %.1f с' % (tb, ts, tr, tv), 'T/ref ср %.4f' % np.mean(T[np.isfinite(T)] / ref[ok][np.isfinite(T)]), flush=True)
print('совпадение T:', np.allclose(out[0][0], out[1][0], equal_nan=True), 'V сетки:', np.allclose(out[0][1], out[1][1]))
