"""Диагностика solve8: почему у части стартов нет V*. Запуск: python tests/diag_solve8.py [--layers 1,-1,0]"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from src.algo import solve8 as S
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--layers', default='1,-1'); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    g = S.load_growN(); layers = [float(x) for x in a.layers.split(',')]; atlas, _ = S.build_atlas(layers, a.seed); G = S.make_graph(g, atlas); G.solve()
    X = S.starts(g, 200, 1); Tr, _ = S.t_ref(X); Vl = G.Vlayers(X); Vs = Vl.min(1); bad = np.flatnonzero(~np.isfinite(Vs)); print('без V*: %d из %d' % (len(bad), len(X)))
    fin = G.V < S.BIG / 2; print('достижимых узлов по слоям:', {u: round(float(fin[G.layer == i].mean()), 3) for i, u in enumerate(G.us)})
    rows = []
    for i in bad:
        r = dict(x=[round(float(t), 3) for t in X[i]], Tref=round(float(Tr[i]), 2), per_layer={})
        for li, u in enumerate(G.us):
            pi, verts, w, rank = G.locate(u, X[i:i + 1])
            r['per_layer'][str(u)] = 'нет ячейки' if not len(pi) else ('ячеек %d, вершин с BIG: %s' % (len(pi), [int((G.V[v] >= S.BIG / 2).sum()) for v in verts]))
        rows.append(r)
    why = {}
    for r in rows:
        k = 'нет ячейки ни в одном слое' if all(v == 'нет ячейки' for v in r['per_layer'].values()) else 'ячейка есть, вершины BIG (нет пути к цели)'; why[k] = why.get(k, 0) + 1
    print(why)
    for r in sorted(rows, key=lambda r: -r['Tref'])[:5]: print(r)
    # покрытие геометрией: доля стартов, лежащих хоть в одной ячейке
    cov = np.array([any(len(G.locate(u, X[i:i + 1])[0]) for u in G.us) for i in range(len(X))]); print('в какой-либо ячейке: %.3f' % cov.mean())
    # оптимальный путь стартов выходит за поле? (max |x| вдоль ref-траектории; поле XL = %g, клетки до XL+GM)
    from src.algo import ref_di_disk as R
    _, info = R.T_ref(X, float(g.RHOV[0]), return_details=True); mx = []
    for x, (s, t1, tot) in zip(X, info):
        t1_ = tot if t1 is None else t1; xs = []
        for tt in np.linspace(0, tot, 300): a = min(tt, t1_); b = tt - a; x1 = x[0] + x[1] * a + s * a * a / 2; v1 = x[1] + s * a; xs.append(x1 + v1 * b - s * b * b / 2)
        mx.append(np.abs(xs).max())
    mx = np.array(mx); lim = float(g.XLV[0]) + g.GM; print('оптимум выходит за поле+GM (%g): %d стартов; без V*: %d, из них с выходом: %d' % (lim, (mx > lim).sum(), len(bad), (mx[bad] > lim).sum()))
    print('max|x| по оптимуму у стартов без V*: мин %.2f' % mx[bad].min() if len(bad) else '')
