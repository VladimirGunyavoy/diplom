"""b3: из каких вершин ячейки берётся V*(старт) — для выбранных стартов бенча. python pend_cell.py 50 28"""
import os, sys, numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); PI = np.pi; rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1)
ref = np.load('pend_ref_best_u0.3.npy'); atlas, tg = S.build_atlas(list(g.US), 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
for i in map(int, sys.argv[1:]):
    X = Q[i:i + 1]; VL = G.Vlayers(X)[0]; print('RES старт %d (%.3f,%.3f) T_ref %.2f V по слоям %s' % (i, *Q[i], ref[i], np.round(VL, 2)))
    for li, u in enumerate(G.us):
        pi, verts, w, rank = G.locate(u, X)
        for j in range(len(pi)):
            v = verts[j]; print('RES   u=%+.1f rank %d w=%s узлы: %s  P=%s  V=%s  Vинт=%.2f' % (u, rank[j], np.round(w[j], 2), v.tolist(), [tuple(np.round(G.P[k], 2)) for k in v], np.round(G.V[v], 2), (w[j] * G.V[v]).sum()))
