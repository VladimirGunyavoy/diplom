"""H1 дерево цепочек на маятнике u=.3, τ=.25: V(старт) против мелкой V (сетка 252×241) и грубых сеток; N спор. Запуск из v6 (~несколько минут)."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_tree import *
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
u = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3; tau = 0.25; Rg = 0.5; f = pendulum(1.0)
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
S = Sys(fl, 2, (1.0, 1.0), lambda P: np.where(gd(P) < Rg, 0.0, np.nan), lambda P: gd(P) < Rg, [(np.pi, 0.0)], per=2 * np.pi, lim=4.0)
t0 = time.time(); F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500); print('мелкая V %.0f с' % (time.time() - t0), flush=True)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
Vf = [F.value(x) for x in Q]; print('V мелкая на запросах:', np.round(Vf, 2))
for rho, cd in ((0.15, 1.0), (0.08, 1.0), (0.08, 2.0), (0.04, 1.0)):
  for ex in (0.3,):
    r = []; Ns = []
    for x, v in zip(Q, Vf):
        P, sc = build_until_junction(S, tau, x, rho, dmax=cd, extra=ex); Ns.append(len(P)); r.append(sc.value(x) / v)
    print('без T_max rho %.2f dmax %.1f +%.0f%%: N~%d mean %.3f max %.3f' % (rho, cd, ex * 100, np.mean(Ns), np.mean(r), np.max(r)), np.round(r, 2), np.array(Ns), flush=True)
