"""lr4: профиль Q слоёв поперёк v=0 справа от цели (где look застревает): Q_u(x, v) против T_ref. env M, LAYERS. Запуск из v8."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS
from src.algo import ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N')
V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; np.set_printoptions(linewidth=250, precision=3, suppress=True)
vs = np.array([-.15, -.1, -.06, -.04, -.02, -.01, 0, .01, .02, .04, .06, .1, .15])
for x in (.25, .3, .35, .5, -.3):
    X = np.stack([np.full(len(vs), x), vs], 1); Vl = G.Vlayers(X); Tr = R.T_ref(X, G.rho)
    print('RES x=%g M=%d  v      :' % (x, G.M), vs)
    print('RES x=%g T_ref        :' % x, Tr)
    for li, u in enumerate(G.us): print('RES x=%g Q[u=%+g]     :' % (x, u), Vl[:, li])
    print('RES x=%g V*−T_ref     :' % x, Vl.min(1) - Tr, flush=True)
