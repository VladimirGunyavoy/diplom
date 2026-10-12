"""W2: складка v=0 у цели — Q_{-1}(x, v) по v и V*−T_ref на x .25–.5. env M, XENTRY (горизонт точного входа; 0 — выкл.), LAYERS. Запуск из v8."""
import os, sys
import numpy as np
os.environ.setdefault('M', '5'); sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
np.set_printoptions(linewidth=250, precision=3, suppress=True); vs = np.array([-.15, -.1, -.06, -.04, -.02, 0, .02, .04, .06, .1, .15]); li = G.us.index(-1.); worst = 0.; viol = 0; tot = 0; mono = 0
print('RES M=%d XENTRY=%g' % (G.M, G.xentry))
for x in (.25, .3, .35, .4, .5):
    X = np.stack([np.full(len(vs), x), vs], 1); Vl = G.Vlayers(X); Tr = R.T_ref(X, G.rho); e = Vl.min(1) - Tr; q = Vl[:, li]; d = np.diff(q); mono += int((d < -1e-3).sum()); worst = max(worst, np.abs(e).max()); tot += 1
    print('RES x=%.2f V*-T_ref: %s | Q_-1: %s | Q_-1 немонот. шагов: %d' % (x, e, q, int((d < -1e-3).sum())))
print('RES ИТОГ: max|V*−T_ref| на x .25–.5, v ±.15 = %.3f; немонотонных шагов Q_-1 по v: %d' % (worst, mono))
