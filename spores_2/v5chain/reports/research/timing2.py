"""research-14: полный цикл (постройка + V + агент + сетка V) при FASTLOC 0 и 2 — время по этапам и совпадение результатов."""
import numpy as np, os, time, sys
import grow_cells2d as G
fl = int(sys.argv[1]); G.FASTLOC = fl; t = time.time(); A = G.Atlas(seed=0); tb = time.time() - t
t = time.time(); A.solve(); ts = time.time() - t; Q, ref = G.starts_ref(); ok = ref > .05
t = time.time(); T, sw, _ = A.rollout(Q[ok]); tr = time.time() - t
gx, gw = np.linspace(-G.XL, G.XL, 161), np.linspace(-G.WL, G.WL, 141); GX, GW = np.meshgrid(gx, gw); t = time.time(); VV = A.vstar(np.c_[GX.ravel(), GW.ravel()]); tv = time.time() - t
np.savez('/tmp/fl%d.npz' % fl, T=T, VV=VV, n=len(A.cells))
print('FASTLOC', fl, 'клеток', len(A.cells), 'постройка %.1f с, V %.1f с, агент %.1f с, сетка V %.1f с' % (tb, ts, tr, tv), 'T/ref ср %.4f' % np.mean(T[np.isfinite(T)] / ref[ok][np.isfinite(T)]), flush=True)
