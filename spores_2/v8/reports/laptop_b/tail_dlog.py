"""b2: разбор хвоста T/T_ref по логу решений (dlog): худшие старты, их первые решения/переключения, Q слоёв, V*−T_ref. env: M=5 SOLVE8_PLUS=1 [GSEED=1]; N=300"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
X = S.starts(g, int(os.environ.get('N', 300)), 1); Tr = R.T_ref(X, G.rho); dl = []; T, sw = G.run_agent(X, 3 * Tr + 2, dlog=dl, commit=True, **{k: v for k, v in S.POLICIES['commit'].items() if k != 'commit'}); r = T / Tr; ok = np.isfinite(T)
print('RES дошли %.3f мед %.4f p90 %.4f max %.2f' % (ok.mean(), np.nanmedian(r), np.nanpercentile(r, 90), np.nanmax(r)))
V0 = G.Vlayers(X).min(1)
for i in np.argsort(-np.where(ok, r, 0))[:6]:
    print('RES старт %d x=(%.3f,%.3f) |x|=%.3f T_ref %.3f T %.3f T/Tref %.2f V*(старт)−T_ref %.3f switches %d' % (i, X[i, 0], X[i, 1], np.linalg.norm(X[i]), Tr[i], T[i], r[i], V0[i] - Tr[i], sw[i]))
    for row in [q for q in dl if q[0] == i][:8]: print('RES    step %d t %.2f x %.3f v %.3f u %+.0f→%+.0f V+ %.3f V- %.3f V0 %.3f sw %d %s' % (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]))
