"""W2: карта вокруг круга цели x∈[−.6,.6], v∈[−.4,.4], шаг .01, по каждому слою: Q_u − T_ref и клетка/ячейка, ответившая на запрос (первая по центральности с конечными вершинами).
Запуск из v8: M=5 python3 tests/fold_map.py (env XENTRY, OUT). Пишет OUT.png и OUT.npz."""
import os, sys
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.algo import solve8 as S, ref_di_disk as R
OUT = os.environ.get('OUT', 'fold_map')
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
xs = np.arange(-.6, .6001, .01); vs = np.arange(-.4, .4001, .01); XX, VV = np.meshgrid(xs, vs); X = np.stack([XX.ravel(), VV.ravel()], 1); Tr = R.T_ref(X, G.rho); out = {}
fig, ax = plt.subplots(2, 3, figsize=(18, 9)); BIG = S.BIG
for li, u in enumerate(G.us):
    pi, verts, w, rank = G.locate(u, X); n = len(X); val = np.full(n, np.inf); cell = np.full(n, -1); row = np.full(n, -1); done = np.zeros(n, bool)
    for r in range(G.K):
        m = rank == r
        if not m.any(): continue
        p_, v_, w_ = pi[m], verts[m], w[m]; Vv = G.V[v_]; ok = (Vv < BIG / 2).all(1) & ~done[p_]; val[p_[ok]] = (w_ * Vv)[ok].sum(1); cell[p_[ok]] = (G.nodecell[v_[:, 0]] - G.gcell[u][0])[ok]; row[p_[ok]] = G.rowid[v_[:, 0]][ok]; done[p_[ok]] = True
    inb = np.linalg.norm(X, axis=1) <= G.rho; val[inb] = 0.; err = val - Tr; out['err_%g' % u] = err.reshape(XX.shape); out['cell_%g' % u] = cell.reshape(XX.shape); out['row_%g' % u] = row.reshape(XX.shape)
    e = np.where(np.isfinite(err), err, np.nan); a = ax[0, li].pcolormesh(xs, vs, e.reshape(XX.shape), cmap='RdBu_r', vmin=-.3, vmax=.3, shading='auto'); plt.colorbar(a, ax=ax[0, li]); ax[0, li].set_title('u=%+g: Q_u − T_ref (синий — занижено, красный — завышено; серое — нет значения)' % u, fontsize=9)
    c = np.where(cell >= 0, cell, np.nan).reshape(XX.shape); a = ax[1, li].pcolormesh(xs, vs, c, cmap='tab20', shading='auto'); ax[1, li].set_title('u=%+g: клетка слоя, ответившая на запрос' % u, fontsize=9)
    for k in (0, 1): ax[k, li].add_patch(plt.Circle((0, 0), G.rho, fill=False, color='k')); ax[k, li].set_aspect('equal'); ax[k, li].set_xlabel('x'); ax[k, li].set_ylabel('v')
    fin = np.isfinite(e) & ~inb; print('RES u=%+g: M=%d XENTRY=%g | вне круга: покрыто %.0f%%, занижено (< −.03) %.1f%%, завышено (> +.03) %.1f%%, мед %.3f, min %.3f max %.3f; клеток отвечало %d' % (u, G.M, G.xentry, 100 * fin.mean(), 100 * (e[fin] < -.03).mean(), 100 * (e[fin] > .03).mean(), np.nanmedian(e[fin]), np.nanmin(e[fin]), np.nanmax(e[fin]), len(np.unique(cell[cell >= 0]))), flush=True)
    # скачки: |ошибка| между соседними по x/v точками > .08
    d = np.abs(np.diff(np.where(np.isfinite(e), e, 0).reshape(XX.shape), axis=1)); print('RES   скачков по x (>.08): %d; по v: %d' % ((d > .08).sum(), (np.abs(np.diff(np.where(np.isfinite(e), e, 0).reshape(XX.shape), axis=0)) > .08).sum()))
Vs = np.stack([np.where(np.isfinite(out['err_%g' % u]), out['err_%g' % u] + Tr.reshape(XX.shape), np.inf) for u in G.us]).min(0); es = (Vs - Tr.reshape(XX.shape)); es[~np.isfinite(es)] = np.nan; out['err_star'] = es; inb = (np.linalg.norm(X, axis=1) <= G.rho).reshape(XX.shape); f = np.isfinite(es) & ~inb
print('RES V*−T_ref вне круга: занижено < −.03: %.1f%%, < −.1: %.2f%%, завышено > +.03: %.1f%%, мед %.3f' % (100 * (es[f] < -.03).mean(), 100 * (es[f] < -.1).mean(), 100 * (es[f] > .03).mean(), np.nanmedian(es[f])))
plt.tight_layout(); plt.savefig(OUT + '.png', dpi=80); np.savez(OUT + '.npz', xs=xs, vs=vs, **out)
