"""laptop-B: порядок посева — первые N клеток слоя: сид p, r, строк, число в очереди. Из v8: MAXC=4 python3 reports/laptop_b/seedorder.py"""
import os, sys
import numpy as np
os.environ.setdefault('MAXC', '4'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S
g = S.load_growN()
for u in (1., -1.):
    cells, idx = g.build_layer(u, np.random.default_rng(0))
    for i, c in enumerate(cells): print('RES u=%+g клетка %d: сид (%.3f, %.3f) r=%.3f nb %d nf %d' % (u, i, *c.p, float(c.r[0]), c.nb, c.nf))
