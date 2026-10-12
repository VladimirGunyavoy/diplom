"""lr3: ошибка V в узлах против T_ref по режимам перехода E3 (lin / lin+цена / maxv) и по числу линий M (env M)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S
from src.algo import ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); G = S.make_graph(g, atlas, 'N'); T = R.T_ref(G.P, G.rho); out = np.linalg.norm(G.P, axis=1) > G.rho
inf_ = (np.abs(G.P) <= 2.5).all(1)
print('RES M=%d узлов %d' % (G.M, G.n))
for mode, pen in (('lin', 0.), ('lin', .02), ('lin', .05), ('maxv', 0.)):
    G.mode, G.pen = mode, pen; G.solve(verbose=False); V = np.where(G.V < S.BIG / 2, G.V, np.nan); e = (V - T)[out & np.isfinite(V)]; ei = (V - T)[out & np.isfinite(V) & inf_]
    print('RES %-5s pen %.2f: конечных %.3f проходов %d | V−T_ref p1/p10/p25/p50/p75/p90/p99: %s | в поле ±2.5: %s | доля < −.05: %.3f, > +.05: %.3f' % (mode, pen, np.isfinite(V).mean(), G.passes, np.round(np.percentile(e, [1, 10, 25, 50, 75, 90, 99]), 3), np.round(np.percentile(ei, [1, 10, 50, 90, 99]), 3), (e < -.05).mean(), (e > .05).mean()))
