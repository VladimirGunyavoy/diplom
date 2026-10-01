"""research hub-research-6: двойной маятник g=1 (2 мотора, |τ_i|≤1) — удержание вверху: вершины 4 / сетка 3×3 (каналы+дрейф+диагонали) / спектр (непрерывная коробка
по тензорной квадратичной интерполяции образа из 9 слоёв). Выбор: argmin x_endᵀ P x_end (LQR). Метрики как в spectrum_hold.py."""
import numpy as np, itertools
from scipy.linalg import solve_continuous_are
exec(open('reports/research/spectrum_image.py').read().split("def errs")[0])      # rk4, dpend
X0 = np.array([np.pi, 0, 0, 0.])
f = lambda d, u: dpend(X0 + d, u)
eps = 1e-6; A = np.array([(f(eps*e, np.zeros(2)) - f(-eps*e, np.zeros(2)))/(2*eps) for e in np.eye(4)]).T
B = np.array([(f(np.zeros(4), eps*e) - f(np.zeros(4), -eps*e))/(2*eps) for e in np.eye(2)]).T
P = solve_continuous_are(A, B, np.eye(4), np.eye(2))
L3 = lambda s, u: {-1: u*(u-1)/2, 0: 1-u*u, 1: u*(u+1)/2}[s]
G = np.linspace(-1, 1, 41); GG = np.array(list(itertools.product(G, G)))
W = np.array([[L3(s0, a)*L3(s1, b) for (s0, s1) in itertools.product((-1, 0, 1), repeat=2)] for a, b in GG])   # (1681, 9)

def run(x0, tau, mode, T=10.0):
    x = np.array(x0, float); t = 0; us = []; xs = []; t_in = np.nan
    while t < T - 1e-9:
        S = list(itertools.product((-1, 0, 1), repeat=2)); E = np.array([rk4(f, x, np.array(s, float), tau, 20) for s in S])
        if mode == 'v4': idx = [i for i, s in enumerate(S) if 0 not in s]; Z = E[idx]; U = np.array(S, float)[idx]
        elif mode == 'g9': Z = E; U = np.array(S, float)
        else: Z = W @ E; U = GG
        k = np.argmin(np.einsum('ni,ij,nj->n', Z, P, Z)); u = U[k]
        x = rk4(f, x, u, tau, 20); t += tau; us.append(u); xs.append(np.abs(x).max())
        if np.isnan(t_in) and xs[-1] < .01: t_in = t
        if xs[-1] > 3: return np.nan, np.inf, np.nan
    xs = np.array(xs); n5 = int(3/tau); us = np.array(us)
    return t_in, xs[-n5:].max(), np.abs(np.diff(us, axis=0)).sum()/T

rng = np.random.default_rng(2); D0 = rng.uniform(-.1, .1, (6, 4))
for tau in (.05, .1, .2):
    for mode in ('v4', 'g9', 'spec'):
        r = np.array([run(d, tau, mode) for d in D0]); ok = ~np.isnan(r[:, 0])
        print(f'τ={tau:<4} {mode:4}  дошли до .01: {ok.sum()}/6  остаток max {r[:,1].max():.1e}  TV(u)/с мед {np.nanmedian(r[:,2]):.2f}', flush=True)
