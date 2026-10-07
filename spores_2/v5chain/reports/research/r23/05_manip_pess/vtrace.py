"""research-23: жадный спуск по V* (A_g50d1p1, manip) от стартов 3, 5, 19: u = argmin DTN + V*(step(y, u)); печать V*(y_t) + t (должна быть ≈ const) и где она прыгает вверх."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open(os.environ.get('AF', 'A_g50d1p1.pkl'), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']
R = np.load(os.path.expanduser('~/spore_v5/wb4/spores_2/v5chain/reports/research/r23/manip_ref_ocp_20.npy'))
for q in (3, 5, 19):
    y = R[q, :4].copy(); t = 0.; v0 = A.vstar(y[None])[0]; print('== старт %d, эталон %.3f, V*0 %.3f' % (q, R[q, 4], v0), flush=True); prev = v0
    for k in range(80):
        Y = np.array([G.step(y, u) for u in G.US]); vs = A.vstar(Y); j = int(np.argmin(vs)); y = Y[j]; t += G.DTN; v = vs[j]
        jump = v + t - (prev + t - G.DTN) - 0. ; prev = v
        if k % 5 == 0 or abs(v + t - v0) > .15 * v0 or v < .05:
            print('t %.1f u%d V* %.3f V*+t %.3f | y %s | goal_dist %.2f' % (t, j, v, v + t, np.round(G.wrapy(y), 2).tolist(), G.goal_dist(y[None])[0]), flush=True)
        if G.ingoal(y[None])[0] or v >= G.BIG / 2: print('стоп: в цели' if v < G.BIG / 2 else 'стоп: V = BIG', 't %.1f' % t, flush=True); break
