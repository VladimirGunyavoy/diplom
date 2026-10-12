"""b2: регрессия WARM/tol: V узлов и дошедшие, WARM=0 (tol 1e-10) против WARM=1 (tol 1e-6). env как окно: M=5 MAXC=3 SOLVE8_PLUS=1 [LAYERS=1,-1]"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')], 0, par=False); res = {}
VAR = [(0, 1e-10), (1, 1e-6), (0, 1e-6), (1, 1e-10)]
for w, tl in VAR:
    os.environ['WARM'] = str(w); os.environ['WARM_TOL'] = str(tl); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); t0 = time.perf_counter(); G.solve(verbose=False, tol=(tl if not w else 1e-10)); ts = time.perf_counter() - t0
    X = S.starts(g, 300, 1); Tr = R.T_ref(X, G.rho); T, sw = G.run_agent(X, 3 * Tr + 2, **S.POLICIES['commit']); ok = np.isfinite(T)
    res[(w, tl)] = (G.Vn.copy(), G.V.copy(), G.n); print('RES WARM=%d tol=%g solve %.1f с проходов(посл) %d conv %s узлов %d | конечных V %.4f | дошли %d/300 T/T_ref мед %.4f' % (w, tl, ts, G.passes, G.converged, G.n, (G.Vn < S.BIG / 2).mean(), ok.sum(), np.median(T[ok] / Tr[ok])), flush=True)
a = res[VAR[0]]
for key in VAR[1:]:
    b = res[key]; print('RES vs', key)
    if a[2] != b[2]: print('RES разное число узлов', a[2], b[2]); continue
    for nm, x, y in (('Vn', a[0], b[0]), ('Q', a[1], b[1])):
        fa, fb = x < S.BIG / 2, y < S.BIG / 2; both = fa & fb; d = np.abs(x[both] - y[both]); print('RES   %s: другая конечность %d, max|Δ| %.2e p99 %.2e доля Δ>1e-3: %.4f' % (nm, (fa != fb).sum(), d.max(), np.percentile(d, 99), (d > 1e-3).mean()))
