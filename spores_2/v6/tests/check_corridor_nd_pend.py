"""Коридор на адаптивном дереве, маятник u=.3 (цель верх, окно Rg=.5): T коридора / мелкая V. Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
u = 0.3; tau = 0.25; Rg = 0.5; rho = 0.05; f = pendulum(1.0)
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t, dt_max=0.05)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
g = lambda x: np.array([Rg ** 2 - gd(x) ** 2])
KK = int(sys.argv[1]) if len(sys.argv) > 1 else 5
miss = lambda X: np.maximum(gd(X) - Rg, 0)
S = SysN(fl, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, [(np.pi + Rg * 0.999 * np.cos(a), Rg * 0.999 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)], per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
for NB, NF in ((400, 50),):
    back = build_back(S, tau, NB, rho); r = []; t0 = time.time()
    for x in Q:
        b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=KK); r.append(np.nan if b is None else b[0] / F.value(x))
    print('NB %d NF %d: %s mean %.3f (%.0f с)' % (NB, NF, np.round(r, 2), np.nanmean(r), time.time() - t0), flush=True)
