"""research-23: на точках жадного спуска от старта 3 (A_g50d1p1): V*(y) (интерполянт) против Беллмана B(y) = min_u DTN + V*(step(y,u)); у min-стенсила — размах V по вершинам (max − min, с весом > .02) и сумма весов: излом внутри гиперячейки ⇒ большой размах и V* < B."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open('A_g50d1p1.pkl', 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']
R = np.load(os.path.expanduser('~/spore_v5/wb4/spores_2/v5chain/reports/research/r23/manip_ref_ocp_20.npy'))
def detail(y):
    I, IDX, W = A.stencils(y[None]); VI = A.V[IDX]; v = A.interp(W, VI); j = int(np.argmin(v)); m = W[j] > .02
    return v[j], VI[j][m].min(), VI[j][m].max(), len(I)
for q in (3, 19):
    y = R[q, :4].copy(); t = 0.; print('== старт', q, flush=True)
    for k in range(25):
        Y = np.array([G.step(y, u) for u in G.US]); vs = A.vstar(Y); B = G.DTN + vs.min(); v, lo, hi, ns = detail(y)
        if k % 2 == 0: print('t %.1f | V* %.3f  B %.3f  B − V* %+.3f | вершины min-стенсила: V от %.2f до %.2f (размах %.2f) | стенсилов %d' % (t, v, B, B - v, lo, hi, hi - lo, ns), flush=True)
        y = Y[int(np.argmin(vs))]; t += G.DTN
