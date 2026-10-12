"""W2: какая клетка/ячейка отвечает на запрос Q_{-1}(x, v) у цели при v<0 (складка на v=0). Запуск из v8: M=5 python3 tests/fold_cell.py. env LAYERS (1,-1,0), XS (.25,.3,.35,.5)."""
import os, sys
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
np.set_printoptions(linewidth=250, precision=3, suppress=True); u = -1.; li = G.us.index(u); M = G.M
print('RES M=%d, слой u=-1: клеток %d' % (M, len(atlas[u])))
for ci, c in enumerate(atlas[u]):
    Gc = np.asarray(c.G); print('RES клетка %d: строк %d, tau 0..%.2f, hb %d nb %d nf %d, r %.3f; центр. столбец: старт (%.3f, %.3f) конец (%.3f, %.3f); bbox x [%.2f, %.2f] v [%.2f, %.2f]' % (ci, len(Gc), c.tau[-1], c.hb, c.nb, c.nf, float(np.asarray(c.r).ravel()[0]), *Gc[0, M // 2], *Gc[-1, M // 2], Gc[..., 0].min(), Gc[..., 0].max(), Gc[..., 1].min(), Gc[..., 1].max()))
for x in [float(t) for t in os.environ.get('XS', '.25,.3,.35,.5').split(',')]:
    for v in (-.15, -.1, -.06, -.02, 0., .02, .06, .15):
        X = np.array([[x, v]]); pi, verts, w, rank = G.locate(u, X); Tr = R.T_ref(X, G.rho)[0]
        for r in range(len(pi)):
            vv = verts[r]; cell = G.nodecell[vv[0]] - G.gcell[u][0]; Qv = G.V[vv]
            print('RES x=%.2f v=%+.2f T_ref %.3f | ранг %d клетка %d, строка %d | вершины (x,v,tau,Q): %s | вес %s | Q_инт %.3f' % (x, v, Tr, rank[r], cell, G.rowid[vv[0]], ' '.join('(%.2f,%.2f,%.2f,%.2f)' % (*G.P[k], G.tau[k], G.V[k]) for k in vv), w[r], float((w[r] * Qv).sum())))
