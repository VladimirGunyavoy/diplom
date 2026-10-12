"""lr4: карта ошибки у цели — Q_u(x, v) − T_ref и V* − T_ref на сетке x∈[−.7,.7], v∈[−.45,.45] (шаг .0125), Q lin, слои 1,−1,0. env M. Запуск из v8."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS
from src.algo import ref_di_disk as R
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho; V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q
xs = np.arange(-.7, .7001, .0125); vs = np.arange(-.45, .4501, .0125); XX, VV = np.meshgrid(xs, vs); P = np.stack([XX.ravel(), VV.ravel()], 1); print('RES точек', len(P), flush=True)
Tr = np.concatenate([R.T_ref(P[i:i + 500], rho) for i in range(0, len(P), 500)]); Vl = G.Vlayers(P); Vl[~np.isfinite(Vl)] = np.nan; out = np.linalg.norm(P, axis=1) > rho
np.savez('goal_map_M%d.npz' % G.M, P=P, Tr=Tr, Vl=Vl); E = np.nanmin(Vl, 1) - Tr; e = E[out & np.isfinite(E)]
print('RES M=%d V*−T_ref у цели: мед %+.3f p1 %+.3f p10 %+.3f p90 %+.3f p99 %+.3f; <−.05: %.3f, >+.05: %.3f; не покрыто %.3f' % (G.M, np.median(e), *np.percentile(e, [1, 10, 90, 99]), (e < -.05).mean(), (e > .05).mean(), (~np.isfinite(E) & out).mean()), flush=True)
fig, ax = plt.subplots(1, 4, figsize=(22, 4.6)); th = np.linspace(0, 2 * np.pi, 200)
for a, (ttl, Z) in zip(ax, [('V* − T_ref', E)] + [('Q[u=%+g] − T_ref' % u, Vl[:, li] - Tr) for li, u in enumerate(G.us)]):
    Z = np.where(out, Z, np.nan).reshape(XX.shape); im = a.pcolormesh(XX, VV, Z, cmap='RdBu_r', vmin=-.2, vmax=.2, shading='auto'); a.plot(rho * np.cos(th), rho * np.sin(th), 'k-'); a.plot(G.P[:, 0], G.P[:, 1], 'k.', ms=1.5, alpha=.5); a.set_xlim(xs[0], xs[-1]); a.set_ylim(vs[0], vs[-1]); a.set_title(ttl + ' (M=%d; синее — занижено, красное — завышено; точки — узлы)' % G.M, fontsize=8); a.set_xlabel('x'); a.set_ylabel('v'); fig.colorbar(im, ax=a)
fig.tight_layout(); fig.savefig('goal_map_M%d.png' % G.M, dpi=110); print('RES картинка', flush=True)
