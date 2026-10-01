"""research hub-research-4: маятник θ̈ = −sin θ + u, |u| ≤ U, из покоя внизу — «энергонакачка bang-bang + LQR» vs оптимум по времени
(bang-bang с N дугами, длительности SLSQP, конец |(φ, ω)| ≤ EPS у верха). Запуск: python3 pend_energy_vs_opt.py [U]"""
import sys, numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
U = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3; EPS = 0.05; H = 0.002
f = lambda x, u: np.array([x[1], -np.sin(x[0]) + u])
def step(x, u, h=H):
    k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
wr = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
dist = lambda x: np.hypot(wr(x[0] - np.pi), x[1])
A = np.array([[0, 1], [1, 0.]]); Bm = np.array([[0], [1.]]); P = solve_continuous_are(A, Bm, np.eye(2), np.array([[10.]])); K = (Bm.T @ P / 10.)[0]
def lqr_ok(x, T=15):
    for _ in range(int(T / H)):
        x = step(x, float(np.clip(-K @ np.array([wr(x[0] - np.pi), x[1]]), -U, U)))
        if dist(x) < EPS: return True
    return False
# 1) энергонакачка + LQR: качаем u = U·sign(ω), пока LQR (насыщенный) из текущей точки не доводит до цели (проверка раз в 0.05 с)
x = np.array([0.0, 0.0]); t = 0.0; arcs = []; cur = U; d0 = 0.0; mode = 'pump'
while t < 60 and dist(x) >= EPS:
    if mode == 'pump':
        if int(round(t / H)) % 25 == 0 and lqr_ok(x.copy()): mode = 'lqr'; arcs.append((cur, t - d0))
        else:
            u = U if x[1] >= 0 else -U
            if u != cur: arcs.append((cur, t - d0)); d0 = t; cur = u
            x = step(x, u); t += H; continue
    x = step(x, float(np.clip(-K @ np.array([wr(x[0] - np.pi), x[1]]), -U, U))); t += H
T_heur = t; print('энергонакачка + LQR: T = %.2f, дуг накачки %d, длительности %s' % (T_heur, len(arcs), [round(d, 2) for _, d in arcs]))
# 2) оптимум: bang-bang N дуг, первая u0 = ±U, знаки чередуются; SLSQP по длительностям; конец в шаре EPS
def end(d, u0, h=0.01):
    x = np.array([0.0, 0.0]); u = u0
    for di in d:
        n = max(1, int(np.ceil(di / h))); hh = di / n
        for _ in range(n): x = step(x, u, hh)
        u = -u
    return x
best = None
for N in range(2, 9):
    for u0 in (U, -U):
        for seed in range(6):
            rng = np.random.default_rng(seed); d0 = rng.uniform(0.8, 3.5, N)
            if seed == 0 and u0 == U and len(arcs) + 1 >= N: d0 = np.array([d for _, d in arcs][:N - 1] + [1.0])[:N]
            cons = {'type': 'ineq', 'fun': lambda d, u0=u0: EPS ** 2 - wr(end(d, u0)[0] - np.pi) ** 2 - end(d, u0)[1] ** 2}
            r = minimize(lambda d: d.sum(), d0, method='SLSQP', bounds=[(0, 8)] * N, constraints=[cons], options={'maxiter': 200})
            if r.success and dist(end(r.x, u0, 0.002)) < EPS * 1.05 and (best is None or r.x.sum() < best[0]):
                best = (r.x.sum(), N, u0, np.round(r.x, 2))
    print('N=%d: лучший пока %s' % (N, None if best is None else '%.3f (N=%d, u0=%+.1f, дуги %s)' % best), flush=True)
print('ИТОГ U=%.2f: энергонакачка+LQR %.2f с, оптимум bang-bang %.2f с, отношение %.3f' % (U, T_heur, best[0], T_heur / best[0]))
