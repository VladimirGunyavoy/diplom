"""research-23: вес вершин-узлов ЦЕЛИ (V = 0) в стенсилах точек вне цели — по траектории спуска от старта 3 (как vtrace) и по 20k случайных точек у цели (goal_dist .05–.6): гипотеза «ноль цели размазан интерполяцией наружу»."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open(os.environ.get('AF', 'A_g50d1p1.pkl'), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']
def gw(Y):
    I, IDX, W = A.stencils(Y); v = A.interp(W, A.V[IDX]); wg = (W * A.goal[IDX]).sum(1); out = np.full(len(Y), np.nan); best = np.full(len(Y), np.inf)
    for j in range(len(I)):
        if v[j] < best[I[j]]: best[I[j]] = v[j]; out[I[j]] = wg[j]
    return out, best
rng = np.random.default_rng(3); Y = np.c_[rng.uniform(-1, 1, (60000, 2)), rng.uniform(-1.2, 1.2, (60000, 2))]; d = G.goal_dist(Y); Y = Y[(d > .05) & (d < .6) & ~G.ingoal(Y)][:20000]; d = G.goal_dist(Y)
w, v = gw(Y); ok = np.isfinite(v) & (v < G.BIG / 2)
for a, b in ((.05, .15), (.15, .3), (.3, .45), (.45, .6)):
    m = ok & (d >= a) & (d < b); print('goal_dist %.2f–%.2f: точек %d | вес вершин цели у min-стенсила: мед %.3f, доля > .1: %.2f | V* мед %.2f' % (a, b, m.sum(), np.median(w[m]), (w[m] > .1).mean(), np.median(v[m])), flush=True)
