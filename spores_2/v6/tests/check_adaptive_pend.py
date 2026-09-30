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
for n_th, n_w in ((24, 21), (48, 41), (96, 81)):
    G = PendAtlas(n_th=n_th, n_w=n_w, wmax=4.0, tau=tau, umax=u, R_goal=Rg); G.solve(iters=1500); r = [G.value(x) / v for x, v in zip(Q, Vf)]
    print('сетка %dx%d N=%d: V/Vfine mean %.3f max %.3f' % (n_th, n_w, n_th * n_w, np.mean(r), np.max(r)), flush=True)
for N in (100, 200, 400, 800):
    r = []; Ns = []; t0 = time.time()
    for x, v in zip(Q, Vf):
        P = build_tree(S, tau, [x], N, 0.15, sw_w=10.0, T_max=1.3 * v + 1); sc = Scattered(S, P, tau); sc.solve(); Ns.append(len(P)); r.append(sc.value(x) / v)
    print('дерево N<=%d: N~%d V/Vfine mean %.3f max %.3f (%.0f с)' % (N, np.mean(Ns), np.mean(r), np.max(r), time.time() - t0), flush=True)
