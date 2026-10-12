"""laptop-B: мало спор на слой (MAXC=2/3, слои 1,-1): доля узлов с конечными V/Q по клеткам, calc 'v' и 'q'. Из v8: MAXC=2 python3 reports/laptop_b/few_cells.py"""
import os, sys
import numpy as np
os.environ.setdefault('MAXC', '2'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
print('RES MAXC=%s клеток %s узлов %d' % (os.environ['MAXC'], {u: len(atlas[u]) for u in atlas}, G.n))
for calc in ('v', 'q'):
    G.set_calc(calc); G.solve(verbose=False); Vn = G.Vn; Q = G.V
    for li, u in enumerate(G.us):
        for ci in range(len(atlas[u])):
            m = (G.layer == li) & (G.nodecell - G.gcell[u][0] == ci); out = m & (np.linalg.norm(G.P, axis=1) > rho)
            print('RES calc=%s u=%+g клетка %d: узлов %d, конечных V %.0f%%, конечных Q(=s.V) %.0f%%' % (calc, u, ci, m.sum(), 100 * (Vn[m] < S.BIG / 2).mean(), 100 * (Q[m] < S.BIG / 2).mean()))
# E3-пары: у узлов каждой клетки — есть ли ячейка чужого слоя (любая / со всеми конечными вершинами по Q) и сколько узлов вообще получили конечное Q
G.set_calc('q'); G.solve(verbose=False); Q = G.V
for li, u in enumerate(G.us):
    for ci in range(len(atlas[u])):
        m = (G.layer == li) & (G.nodecell - G.gcell[u][0] == ci); row = []
        for li2, pairs in enumerate(G.e3):
            if li2 == li: continue
            pi, verts, w, rank = pairs; inm = m[pi]; anyp = np.unique(pi[inm]); fin = (Q[verts] < S.BIG / 2).all(1) & inm; finp = np.unique(pi[fin])
            row.append('→слой %+g: есть ячейка у %.0f%% узлов, из них все вершины с конечным Q у %.0f%%' % (G.us[li2], 100 * len(anyp) / m.sum(), 100 * len(finp) / m.sum()))
        print('RES E3 u=%+g клетка %d: %s' % (u, ci, '; '.join(row)))
