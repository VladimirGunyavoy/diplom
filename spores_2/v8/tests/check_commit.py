"""W1 (laptop-v5chain-worker-6): штатный агент commit (solve8.Graph.run_commit) на Q lin, слои LAYERS (1,-1,0), M (5), N стартов (1000, сид 1).
Критерий: в поле дошли 100%, T/T_ref мед ≤ 1.01, p90 ≤ 1.05 (эталон — v5chain reports/research/lr4/stuck0_M5.log). Время: look по V (старое) и commit по Q (новое) на одних стартах.
Запуск из v8: M=5 python3 tests/check_commit.py"""
import os, sys, time
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N')
X0 = S.starts(g, int(os.environ.get('N', '1000')), 1); Tr = R.T_ref(X0, G.rho); mx = S.path_extent(X0, G.rho); inw = mx <= float(g.XLV[0]); tmax = 3 * Tr + 2
def rep(name, pol):
    t1 = time.perf_counter(); T, sw = G.run_agent(X0, tmax, **S.POLICIES[pol]); dt = time.perf_counter() - t1; ok = np.isfinite(T); r = T[ok & inw] / Tr[ok & inw]
    print('RES %-18s дошли %.3f, в поле %.3f | T/T_ref в поле мед %.3f p90 %.3f max %.2f | перекл ср %.1f | счёт %.1f с (%d стартов)' % (name, ok.mean(), ok[inw].mean(), np.median(r), np.percentile(r, 90), r.max(), sw[inw].mean(), dt, len(X0)), flush=True)
    return ok[inw].mean() == 1 and np.median(r) <= 1.01 and np.percentile(r, 90) <= 1.05
G.solve(verbose=False)
if not os.environ.get('SKIP_OLD'): rep('V old, look', 'look')
G.set_calc('q'); t1 = time.perf_counter(); G.solve(verbose=False); print('RES solve Q lin %.1f с' % (time.perf_counter() - t1), flush=True)
rep('Q lin, look', 'look'); okc = rep('Q lin, commit', 'commit')
for k in ((2, 3, 5) if not os.environ.get('SKIP_OLD') else ()):
    S.COMMIT_EVERY = k; rep('Q lin, commit /%d' % k, 'commit')
print('RES КРИТЕРИЙ W1:', 'OK' if okc else 'НЕ ВЫПОЛНЕН')
