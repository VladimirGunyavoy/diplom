"""laptop-B W2: карта вокруг цели — какая клетка слоя отвечает на запрос (ячейка с макс. центральностью), размеры клетки, ошибка Q_u−T_ref.
Запуск из v8: M=5 python3 reports/laptop_b/cellmap.py  (env M, CALC=q). Вывод: cellmap_M*.png, .npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S, ref_di_disk as R
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc(os.environ.get('CALC', 'q')); G.solve(verbose=True); rho = G.rho
xs = np.arange(-.7, .7001, .0125); vs = np.arange(-.45, .4501, .0125); XX, VV = np.meshgrid(xs, vs); P = np.stack([XX.ravel(), VV.ravel()], 1)
Tr = np.concatenate([R.T_ref(P[i:i + 500], rho) for i in range(0, len(P), 500)]); out = np.linalg.norm(P, axis=1) > rho
Vl = G.Vlayers(P); fig, ax = plt.subplots(2, 3, figsize=(20, 10)); th = np.linspace(0, 2 * np.pi, 200); info = {}
for li, u in enumerate(G.us):
    pi, verts, w, rank = G.locate(u, P); m = rank == 0; cell = np.full(len(P), -1); cell[pi[m]] = G.nodecell[verts[m, 0]]
    ext = np.full(len(P), np.nan); a = G.P[verts[m, 0]]; b = G.P[verts[m, 3]]; ext[pi[m]] = np.linalg.norm(a - b, axis=1)      # диагональ ячейки
    rowid = np.full(len(P), -1); rowid[pi[m]] = G.rowid[verts[m, 0]]
    info[u] = (cell, ext, rowid)
    for r, (Z, ttl, cm, lim) in enumerate([(cell % 17, 'слой %+g: номер клетки (mod 17)' % u, 'tab20', None), (np.where(out, Vl[:, li] - Tr, np.nan), 'слой %+g: Q−T_ref' % u, 'RdBu_r', .2)]):
        z = np.where(cell >= 0, Z, np.nan) if r == 0 else Z; kw = dict(vmin=-lim, vmax=lim) if lim else {}
        im = ax[r, li].pcolormesh(XX, VV, z.reshape(XX.shape), cmap=cm, shading='auto', **kw); ax[r, li].plot(rho * np.cos(th), rho * np.sin(th), 'k-'); ax[r, li].set_title(ttl)
        sel = np.isin(np.arange(G.n), []) 
fig.tight_layout(); fig.savefig('reports/laptop_b/cellmap_M%d.png' % G.M, dpi=80)
np.savez('reports/laptop_b/cellmap_M%d.npz' % G.M, P=P, Tr=Tr, Vl=Vl, **{('cell%+g' % u): info[u][0] for u in info}, **{('ext%+g' % u): info[u][1] for u in info})
for u in G.us:
    c = info[u][0]; q = (np.abs(P[:, 0] - .3) < .15) & (np.abs(P[:, 1]) < .15)
    print('RES u=%+g: клеток, отвечающих в окне x.15–.45 |v|<.15: %s' % (u, np.unique(c[q])), flush=True)
