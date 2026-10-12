"""b2: таблица Q + lookup в агенте: 300 стартов rng 1, commit every=3; точный против таблицы (h=.02,.01). env как bench_opt (MAXC=3 SOLVE8_PLUS=1 или полный атлас)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); t0 = time.perf_counter(); G.solve(verbose=False); print('RES solve %.1f с' % (time.perf_counter() - t0), flush=True)
X0 = S.starts(g, 300, 1); Tr = R.T_ref(X0, G.rho); tm = 3 * Tr + 2; mx = S.path_extent(X0, G.rho); inw = mx <= float(g.XLV[0])
def rep(tag, **k):
    t0 = time.perf_counter(); T, sw = G.run_commit(X0, tm, every=3, **k); dt = time.perf_counter() - t0; ok = np.isfinite(T); r = T[ok & inw] / Tr[ok & inw]; print('RES %-22s в поле дошли %.3f T/Tref мед %.4f p90 %.4f max %.2f sw %.1f | %.3f с' % (tag, ok[inw].mean(), np.median(r), np.percentile(r, 90), r.max(), sw[inw].mean(), dt), flush=True)
rep('точный')
for h in (.02, .01):
    tb = G.build_table(h); print('RES таблица h=%g построение %.1f с, %.0f МБ' % (h, tb, G.tab[0].nbytes / 1e6), flush=True); rep('таблица h=%g' % h)
