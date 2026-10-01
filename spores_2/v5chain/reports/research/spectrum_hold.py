"""research hub-research-6: удержание у цели (маятник вверху, DI в нуле). Шаг τ, u постоянно на шаге; выбор u = argmin x_endᵀ P x_end (P — Риккати LQR).
Варианты: bb2 {±umax}, bb3 {±umax, 0}, spec — u ∈ [−umax, umax] по образу U (3 слоя, квадр. интерполяция). Метрики: время до |x|<0.01, остаточная амплитуда (max|x| за последние 5 с из 15), TV(u)/с."""
import numpy as np
from scipy.linalg import solve_continuous_are

def rk4(f, x, u, t, n=20):
    h = t / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h/2*k1, u); k3 = f(x + h/2*k2, u); k4 = f(x + h*k3, u); x = x + h/6*(k1+2*k2+2*k3+k4)
    return x
top = lambda x, u: np.array([x[1], np.sin(x[0]) + u])          # θ — отклонение от верха: θ'' = sin θ + u
di = lambda x, u: np.array([x[1], u])
A = np.array([[0, 1], [1, 0.]]); A_di = np.array([[0, 1], [0, 0.]]); B = np.array([[0], [1.]])

def run(f, P, x0, umax, tau, mode, T=15.0):
    x = np.array(x0, float); t = 0; us = []; xs = []; t_in = np.nan
    while t < T - 1e-9:
        L = {k: rk4(f, x, k*umax, tau) for k in (-1, 0, 1)}
        Z = lambda s: L[0] + s*(L[1]-L[-1])/2 + s*s*((L[1]+L[-1])/2 - L[0])
        cand = {'bb2': [-1, 1], 'bb3': [-1, 0, 1], 'spec': np.linspace(-1, 1, 401)}[mode]
        s = min(cand, key=lambda s: Z(s) @ P @ Z(s))
        x = rk4(f, x, s*umax, tau); t += tau; us.append(s*umax); xs.append(np.abs(x).max())
        if np.isnan(t_in) and xs[-1] < .01: t_in = t
    xs = np.array(xs); n5 = int(5/tau)
    return t_in, xs[-n5:].max(), np.sum(np.abs(np.diff(us)))/T

rng = np.random.default_rng(1)
for name, f, AA, umax, R in (('маятник верх u=.3', top, A, .3, .15), ('DI ноль', di, A_di, 1., .5)):
    P = solve_continuous_are(AA, B, np.eye(2), np.array([[1.]]))
    X0 = rng.uniform(-R, R, (10, 2)); print(name)
    for tau in (.05, .1, .25):
        for mode in ('bb2', 'bb3', 'spec'):
            r = np.array([run(f, P, x0, umax, tau, mode) for x0 in X0])
            ok = ~np.isnan(r[:, 0])
            print(f'  τ={tau:<4} {mode:4}  дошли до .01: {ok.sum()}/10 (мед t {np.nanmedian(r[:,0]) if ok.any() else np.nan:.2f})  остаток max {r[:,1].max():.1e}  TV(u)/с мед {np.median(r[:,2]):.2f}')
