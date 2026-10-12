"""laptop-B W2: столбцы споры цели (клетка 0 каждого слоя): минимальное расстояние столбца до начала координат, вход в круг ρ, шаг строк возле цели. Из v8: M=5 python3 reports/laptop_b/goalcols.py"""
import os, sys
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); rho = float(g.RHOV[0]); np.set_printoptions(precision=3, suppress=True, linewidth=200)
for u in atlas:
    c = atlas[u][0]; Gc = np.asarray(c.G); print('RES u=%+g клетка0: строк %d, столбцов %d, r %.3f, hb %d nb %d nf %d' % (u, len(Gc), Gc.shape[1], float(np.asarray(c.r).ravel()[0]), c.hb, c.nb, c.nf))
    for j in range(Gc.shape[1]):
        col = Gc[:, j]; d = np.linalg.norm(col, axis=1); k = d.argmin()
        # мин. расстояние до начала по ломаной (отрезки)
        best = 1e9
        for a, b in zip(col[:-1], col[1:]):
            t = np.clip(-a @ (b - a) / ((b - a) @ (b - a) + 1e-30), 0, 1); best = min(best, np.linalg.norm(a + t * (b - a)))
        print('RES   столбец %d: узел min|.|=%.3f (строка %d), ломаная min=%.4f %s | %s' % (j, d[k], k, best, 'ВХОДИТ' if best < rho else 'МИМО', ' '.join('(%.2f,%.2f)' % tuple(p) for p in col[max(k-1,0):k+2])))
