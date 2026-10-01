import sys; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import di, pend
from src.cells7.hold import HoldCell
def rk4(S, x, u, t, n=20):
    h = t / n
    for _ in range(n):
        k1 = S._f(x, u); k2 = S._f(x + h / 2 * k1, u); k3 = S._f(x + h / 2 * k2, u); k4 = S._f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
def run(S, H, g, x0, mode, T=15.0):
    x = g + x0; t = 0; us = []; res = []; tin = np.nan
    while t < T - 1e-9:
        dx = x - g; dx[0] = (dx[0] + np.pi) % (2 * np.pi) - np.pi if S.per[0] > 0 else dx[0]
        u = H.choose(dx, mode); x = rk4(S, x, u, H.tau); t += H.tau; us.append(u); d = np.abs(x - g); d[0] = min(d[0], abs(d[0] - 2 * np.pi)) if S.per[0] > 0 else d[0]; res.append(d.max())
        if np.isnan(tin) and res[-1] < .01: tin = t
    n5 = int(5 / H.tau); return tin, max(res[-n5:]), np.sum(np.abs(np.diff(us))) / T
rng = np.random.default_rng(1)
for name, S, g, R in (('DI ноль', di(), np.zeros(2), .5), ('маятник верх u=.3', pend(), np.array([np.pi, 0.]), .15)):
    X0 = rng.uniform(-R, R, (10, 2)); print(name)
    for tau in (.05, .1, .25):
        H = HoldCell(S, g, tau)
        for mode in ('bb2', 'bb3', 'spec'):
            r = np.array([run(S, H, g, x0, mode) for x0 in X0]); ok = ~np.isnan(r[:, 0])
            print(f'  τ={tau:<4} {mode:4} до .01: {ok.sum()}/10 остаток max {r[:,1].max():.1e} TV/с мед {np.median(r[:,2]):.2f}')
