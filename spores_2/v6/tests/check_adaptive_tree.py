"""H1 дерево цепочек, DI: N и V(старт)/T* против решётки. Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_tree import *
from src.atlas6.cell import flow
from src.atlas6.agent import T_star
tau = 0.25; Rg = 0.25
def di(umax=1.0):
    fl = lambda P, s, t: flow(P, umax * (1 if s == 0 else -1), t)
    gv = lambda P: np.where(np.hypot(P[..., 0], P[..., 1]) < Rg, np.vectorize(T_star)(P[..., 0], P[..., 1]), np.nan)
    return Sys(fl, 2, (1.0, 1.0), gv, lambda P: np.hypot(P[..., 0], P[..., 1]) < Rg, [(0, 0)], lim=8.0)
S = di(); rng = np.random.default_rng(0); Q = [rng.uniform(-3, 3, 2) for _ in range(10)]
for h in (1.0, .5, .25):
    xs = np.arange(-6, 6 + 1e-9, h); P = np.array([(x, v) for x in xs for v in xs]); sc = Scattered(S, P, tau); sc.solve()
    r = [sc.value(x0) / T_star(*x0) for x0 in Q]; print('решётка h=%.2f N=%d: mean %.3f max %.3f' % (h, len(P), np.mean(r), np.max(r)), flush=True)
for N in (60, 120, 250, 500):
  for rho in (.1, .2):
    r = []; t0 = time.time(); Ns = []
    for x0 in Q:
        P = build_tree(S, tau, [x0], N, rho); sc = Scattered(S, P, tau); sc.solve(); Ns.append(len(P)); r.append(sc.value(x0) / T_star(*x0))
    print('дерево N<=%d rho=%.2f N~%d: mean %.3f max %.3f (%.0f с)' % (N, rho, np.mean(Ns), np.mean(r), np.max(r), time.time() - t0), flush=True)
