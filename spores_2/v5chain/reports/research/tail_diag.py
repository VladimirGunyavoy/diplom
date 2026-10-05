"""research-14: хвост маятника — где вдоль пути агента рушится обещание V (P(t) = t + V*(x_t) должно быть постоянным). Конфиг — окружением, как compute.py."""
import numpy as np, os, sys, json
import grow_cells2d as G
A = G.Atlas(seed=int(os.environ.get('SEED', 0))); A.solve(); Q, ref = G.starts_ref(); ok = ref > .05; Q, ref = Q[ok], ref[ok]
for k in [int(x) for x in os.environ.get('KS', '61,84,28,5').split(',')]:
    T, sw, path = A.rollout(Q[k:k + 1]); p = path[:, 0]; n = int(np.argmax(np.all(np.abs(p[1:] - p[:-1]) < 1e-12, 1))) or len(p) - 1; p = p[:n + 1]
    V = A.vstar(p); P = np.arange(len(p)) * G.DTN + V; d = np.diff(P); j = int(np.argmax(d))
    print('старт', k, Q[k].round(2), 'ref %.2f V0 %.2f T %.2f' % (ref[k], V[0], T[0]), '| скачки P > .1:', [(i, round(float(d[i]), 2), p[i].round(2).tolist()) for i in np.flatnonzero(d > .1)][:6])
    y = p[j:j + 1]; I, IDX, W = A.stencils(y)
    for idx_, w in zip(IDX, W):
        c = next(c for c in A.cells if c.o <= idx_[0] < c.o + c.G.shape[0] * c.m); vv = A.V[idx_]
        print('   до скачка: клетка u=%+.1f r=%.3f узлы V' % (c.u, c.r), vv.round(2).tolist(), 'вес', w.round(2).tolist(), '→ %.2f' % (w @ vv))
    for u in G.US: print('   шаг u=%+.1f → V* %.2f' % (u, A.vstar(G.step(y, u))[0]))
