"""replay_value на маятнике u=.3 против мелкой V и cross_value (2D-сверка обобщённого nD-кода). Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import *
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
u = 0.3; tau = 0.25; Rg = 0.5; f = pendulum(1.0)
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
S = SysN(fl, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, [(np.pi, 0.0)], per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
Vf = [F.value(x) for x in Q]
for rho in (0.05, 0.03):
  for NB, NF in ((200, 200), (400, 400)):
    t0 = time.time(); back = build_back(S, tau, NB, rho); r = []
    for x, v in zip(Q, Vf):
        V, nb, nf, path = replay_value(S, tau, x, NB, NF, rho, rho, back=back); r.append(V / v)
    print('rho %.2f NB %d NF %d: mean %.3f max %.3f (%.0f с)' % (rho, NB, NF, np.mean(r), np.max(r), time.time() - t0), np.round(r, 2), flush=True)
