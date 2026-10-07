"""research-23: разброс V по стенсилам старта manip (A_g50p1): min (как vstar) / медиана / max / самый центральный (max min(sc, 1−sc), как SONE) / доля веса неизвестных вершин (PESS-досчёт) у min-стенсила."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open('A_g50p1.pkl', 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']
R = np.load(os.path.expanduser('~/spore_v5/wb4/spores_2/v5chain/reports/research/r23/manip_ref_ocp_20.npy')); Q, ref = R[:, :4], R[:, 4]
I, IDX, W = A.stencils(Q); pi, hid, sc = G._pquery(A.qx, G.wrapy(Q)); assert (pi == I).all()
VI = A.V[IDX]; v = A.interp(W, VI); marg = np.minimum(sc, 1 - sc).min(1); unk = (W * (VI >= G.BIG / 2)).sum(1); lay = np.array([0])
for q in range(len(Q)):
    j = np.flatnonzero(I == q); jm = j[np.argmin(v[j])]; jc = j[np.argmax(marg[j])]
    print('старт %2d | /эт: min %.2f мед %.2f max %.2f центр %.2f | у min: вес неизвестных %.2f, запас %.3f' % (q, v[j].min() / ref[q], np.median(v[j]) / ref[q], v[j].max() / ref[q], v[jc] / ref[q], unk[jm], marg[jm]), flush=True)
