"""replay_value_fast == replay_value (маятник u=.3) и скорость. Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import *
from src.atlas6.gcell import rk4, pendulum
u = 0.3; tau = 0.25; Rg = 0.5; f = pendulum(1.0)
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
seeds = [(np.pi + Rg * 0.999 * np.cos(a), Rg * 0.999 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
S = SysN(fl, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, seeds, per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
back = build_back(S, tau, 400, 0.05); rng = np.random.default_rng(1); ta = tb = 0
for _ in range(4):
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    t0 = time.time(); a = replay_value(S, tau, x, 400, 100, .05, .05, back=back); t1 = time.time(); b = replay_value_fast(S, tau, x, 400, 100, .05, .05, back=back); t2 = time.time()
    ta += t1 - t0; tb += t2 - t1; print(a[0], b[0], a[3] == b[3] or 'путь другой'); assert a[0] == b[0] or abs(a[0] - b[0]) < 1e-9
print('replay %.1f с, fast %.1f с' % (ta, tb))
