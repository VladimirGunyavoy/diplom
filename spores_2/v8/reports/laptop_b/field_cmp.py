"""laptop-B: замер рабочей зоны/буфера ДИ. Запуск из v8: M=5 [XL=2.5 VL=2.5 GMX=1] python3 reports/laptop_b/field_cmp.py  (env N стартов=1000, LAYERS)
Печатает: клеток/узлов, время роста и solve Q, доля стартов с V*, в зоне дошли, T/T_ref commit/look (в зоне), сколько стартов выходит за зону и за буфер."""
import os, sys, time
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]
t0 = time.perf_counter(); atlas, _ = S.build_atlas(layers, 0, par=False); tg = time.perf_counter() - t0; G = S.make_graph(g, atlas, 'N')
nc = sum(len(atlas[u]) for u in atlas); G.set_calc('q'); t0 = time.perf_counter(); G.solve(verbose=False); ts = time.perf_counter() - t0
print('RES XLV=%s GMV=%s M=%d | клеток %d (по слоям %s), узлов %d | рост %.1f с, solve Q %.1f с' % (g.XLV, g.GMV, G.M, nc, [len(atlas[u]) for u in atlas], G.n, tg, ts))
X0 = S.starts(g, int(os.environ.get('N', '1000')), 1); Tr = R.T_ref(X0, G.rho); mx = S.path_extent(X0, G.rho); inw = mx <= float(g.XLV[0]); inb = mx <= float(g.XLV[0] + g.GMV[0]); tmax = 3 * Tr + 2
hasV = np.isfinite(G.Vstar(X0)[0])
print('RES стартов %d: оптимум выходит за зону %d, за буфер %d; с V* %.1f%% (в зоне %.1f%%, в буфере %.1f%%)' % (len(X0), (~inw).sum(), (~inb).sum(), 100 * hasV.mean(), 100 * hasV[inw].mean(), 100 * hasV[inb].mean()))
for name, pol in (('Q lin commit', 'commit'), ('Q lin look', 'look')):
    t1 = time.perf_counter(); T, sw = G.run_agent(X0, tmax, **S.POLICIES[pol]); ok = np.isfinite(T); r = T[ok & inw] / Tr[ok & inw]
    print('RES %-13s дошли всего %.3f, в зоне %.3f (не дошло в зоне %d, вне зоны %d) | T/T_ref в зоне мед %.3f p90 %.3f max %.2f | перекл ср %.1f | %.1f с' % (name, ok.mean(), ok[inw].mean(), (~ok & inw).sum(), (~ok & ~inw).sum(), np.median(r), np.percentile(r, 90), r.max(), np.mean(sw[ok]) if np.ndim(sw) else 0, time.perf_counter() - t1), flush=True)
