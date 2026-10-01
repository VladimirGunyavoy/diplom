"""СТАТИСТИКА hub-worker-8: 40 запросов (corridor_batch), маятник u=.3; ref — мелкая V PendAtlas. env NQ, NAME. Коридор на адаптивном дереве, маятник u=.3 (цель верх, окно Rg=.5): T коридора / мелкая V. Запуск из v6."""
import sys, os, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
u = float(os.environ.get('U', 0.3)); tau = 0.25; Rg = 0.5; rho = 0.05; f = pendulum(1.0)
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t, dt_max=0.05)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
g = lambda x: np.array([Rg ** 2 - gd(x) ** 2])
KK = 5
miss = lambda X: np.maximum(gd(X) - Rg, 0)
S = SysN(fl, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, [(np.pi + Rg * 0.999 * np.cos(a), Rg * 0.999 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)], per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500)
import os, json
from src.atlas6.corridor_nd import corridor_batch
NQ = int(os.environ.get('NQ', 40)); rng = np.random.default_rng(1); Q = []
while len(Q) < NQ:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
for NB, NF in ((400, 50), (1000, 150)):
    back = build_back(S, tau, NB, rho); t0 = time.time()
    R_ = corridor_batch(S, fl, Q, back, tau, NF, rho, g, miss, K=KK); tt = time.time() - t0; out = []
    for x, b in zip(Q, R_):
        ok = False
        if b is not None:
            xe = np.array(x, float)
            for s, d in zip(b[1], b[2]): xe = rk4(f, xe, u * (1 if s == 0 else -1), d, dt_max=0.005)
            ok = bool(gd(xe) < Rg + 1e-4)
        out.append(dict(T=None if b is None else float(b[0]), ok=ok, ref=float(F.value(x))))
    r = [o['T'] / o['ref'] for o in out if o['ok']]
    print('NAME pend NB %d NF %d валидно %d/%d T/ref mean %.3f med %.3f max %.3f min %.3f серия %.0f с' % (NB, NF, len(r), NQ, np.mean(r), np.median(r), np.max(r), np.min(r), tt), flush=True)
    json.dump(dict(NB=NB, NF=NF, tsec=tt, res=out), open('reports/stats_pend_%d.json' % NB, 'w'))
