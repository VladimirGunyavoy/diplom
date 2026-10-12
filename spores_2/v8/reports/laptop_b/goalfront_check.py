"""laptop-B: проверка GOALFRONT. Из v8: M=5 [GOALFRONT=0] python3 reports/laptop_b/goalfront_check.py
(1) узлы споры цели вне круга: V − точный вход по своему потоку (Q lin); (2) ошибка запроса МЕЖДУ узлами: точки y = поток u назад на s от (x_c,0), x_c∈(−ρ,ρ)·.99 (сами входят в круг) против точного времени входа; (3) commit/look в зоне."""
import os, sys, time
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False); rho = G.rho
print('RES GOALFRONT=%s GOALW=%s клеток %d узлов %d' % (os.environ.get('GOALFRONT', 'default'), os.environ.get('GOALW', 'default'), sum(len(atlas[u]) for u in atlas), G.n))
for li, u in enumerate(G.us):
    if u == 0: continue
    m = np.flatnonzero((G.layer == li) & (G.nodecell == G.gcell[u][0]) & (np.linalg.norm(G.P, axis=1) > rho)); te = S.entry_time(G.P[m], u, np.full(len(m), 5.), rho); ok = np.isfinite(te) & (G.V[m] < S.BIG / 2)
    print('RES u=%+g споры цели вне круга: узлов %d, с V %d, |V − точный вход| мед %.4f max %.4f' % (u, len(m), ok.sum(), np.median(np.abs(G.V[m][ok] - te[ok])) if ok.any() else -1, np.abs(G.V[m][ok] - te[ok]).max() if ok.any() else -1))
    xc = np.linspace(-.99, .99, 100) * rho; ss = np.arange(.02, 1.2, .02); XC, SS = np.meshgrid(xc, ss); P = np.stack([(XC + u * SS ** 2 / 2).ravel(), (-u * SS).ravel()], 1); P = P[np.linalg.norm(P, axis=1) > rho]
    pi, verts, w, rank = G.locate(u, P); sel = np.zeros(len(P), bool); r0 = rank == 0; sel[pi[r0]] = G.nodecell[verts[r0, 0]] == G.gcell[u][0]; P = P[sel]          # только точки, на которые отвечает спора цели
    te = S.entry_time(P, u, np.full(len(P), 5.), rho); q = G.Vlayers(P)[:, li]; ok = np.isfinite(te) & np.isfinite(q); e = q[ok] - te[ok]
    print('RES u=%+g запрос между узлами в споре цели (%d точек, покрыто %.0f%%): ошибка мед %.3f p90(|.|) %.3f max |.| %.3f, занижено < −.03: %.1f%%' % (u, len(P), 100 * ok.mean(), np.median(e), np.percentile(np.abs(e), 90), np.abs(e).max(), 100 * (e < -.03).mean()))
X0 = S.starts(g, 1000, 1); Tr = R.T_ref(X0, G.rho); inw = S.path_extent(X0, G.rho) <= float(g.XLV[0]); tmax = 3 * Tr + 2
for name, pol in (('commit', 'commit'), ('look', 'look')):
    T, sw = G.run_agent(X0, tmax, **S.POLICIES[pol]); ok = np.isfinite(T); r = T[ok & inw] / Tr[ok & inw]
    print('RES %-6s в зоне дошли %.3f (не дошло %d) | T/T_ref мед %.3f p90 %.3f max %.2f' % (name, ok[inw].mean(), (~ok & inw).sum(), np.median(r), np.percentile(r, 90), r.max()), flush=True)
