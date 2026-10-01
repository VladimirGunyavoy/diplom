import sys; sys.path.insert(0, '.'); sys.path.insert(0, '../v6/src')
import numpy as np
from src.cells7.holdn import HoldCellN, rk4
from atlas6.manip3dyn import f as fm
def run(H, f, x0, mode, T=12.0):
    x = H.g + x0; t = 0; res = []; us = []; tin = np.nan; u = H.ueq
    while t < T - 1e-9:
        dx = x - H.g; u = H.choose_vertices(dx) if mode == 'bb' else H.choose(dx, u)
        x = rk4(f, x, u, H.tau); t += H.tau; us.append(u); res.append(np.abs(x - H.g).max())
        if np.isnan(tin) and res[-1] < .01: tin = t
    n5 = int(4 / H.tau); return tin, max(res[-n5:]), np.sum(np.abs(np.diff(np.array(us), axis=0))) / T
rng = np.random.default_rng(1)
for gname, gg, goal in (('2 зв., g=.3, q=(-π/2,0) [вниз]', 0.3, [-np.pi/2, 0.0, 0, 0]), ('2 зв., g=.3, q=(π/2,0) [вверх]', 0.3, [np.pi/2, 0.0, 0, 0])):
    f = lambda x, u: fm(x, u, g=gg); print(gname, flush=True)
    for tau in (.05, .1, .25):
        H = HoldCellN(f, goal, 2, tau); print('  u_eq', np.round(H.ueq, 3), end='')
        X0 = rng.uniform(-.15, .15, (6, 4)) * np.array([1, 1, 1, 1])
        for mode in ('bb', 'spec'):
            r = np.array([run(H, f, x0, mode) for x0 in X0]); ok = ~np.isnan(r[:, 0])
            print(f' | τ={tau} {mode}: до .01 {ok.sum()}/6 ост {r[:,1].max():.1e} TV {np.median(r[:,2]):.2f}', end='', flush=True)
        print()
