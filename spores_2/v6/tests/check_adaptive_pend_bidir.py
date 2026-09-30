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
gp = [(np.pi + Rg * c * np.cos(a), Rg * c * np.sin(a)) for c in (0, 1) for a in np.arange(0, 2 * np.pi, np.pi / 4)]
S = Sys(fl, 2, (1.0, u), lambda P: np.where(gd(P) < Rg, 0.0, np.nan), lambda P: gd(P) < Rg, gp, per=2 * np.pi, lim=4.0)
t0 = time.time(); F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500); print('мелкая V %.0f с' % (time.time() - t0), flush=True)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
Vf = [F.value(x) for x in Q]; print('V мелкая на запросах:', np.round(Vf, 2))
for lam in (1.0, 1.5, 2.5):
  for NB, NF in ((100, 50), (200, 100), (400, 200), (800, 400)):
    r = []
    for x, v in zip(Q, Vf):
        V, nb, nf = bidir_value(S, tau, x, NB, NF, 0.25, 0.25, lam=lam); r.append(V / v)
    print('bidir lam %.1f NB %d NF %d: mean %.3f max %.3f' % (lam, NB, NF, np.mean(r), np.max(r)), np.round(r, 2), flush=True)
