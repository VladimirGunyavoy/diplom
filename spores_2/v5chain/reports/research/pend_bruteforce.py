"""Маятник u = .3: независимый эталон для запросов, где коридор дерева > Vfine (q2, q7 из corridor_pend.py): перебор чередующихся
топологий (1…5 сегментов, первый знак ±) × 8 случайных стартов SLSQP (min Σdt, конец в окне). Кто прав — дерево или мелкая V.
Запуск из spores_2/v6: python3 ../v5chain/reports/research/pend_bruteforce.py 2 7"""
import sys, os, json; sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
HERE = os.path.dirname(os.path.abspath(__file__))
u = 0.3; tau = 0.25; Rg = 0.5; f = pendulum(1.0)
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
def end(x, sg, d):
    X = np.array(x, float)
    for k, t in enumerate(d): X = rk4(f, X, u * sg * (-1) ** k, t, dt_max=0.02)
    return X
out = {}
for q in map(int, sys.argv[1:]):
    x = Q[q]; vf = F.value(x); r2 = np.random.default_rng(q); best = (np.inf, None)
    for n in range(1, 6):
        for sg in (1, -1):
            con = {'type': 'ineq', 'fun': lambda d: np.array([Rg ** 2 - dth(end(x, sg, d)[0]) ** 2 - end(x, sg, d)[1] ** 2])}
            for _ in range(8):
                d0 = r2.dirichlet(np.ones(n)) * vf * r2.uniform(0.8, 1.3)
                r = minimize(lambda d: d.sum(), d0, jac=lambda d: np.ones_like(d), bounds=[(0, 2 * vf)] * n, constraints=[con], method='SLSQP', options=dict(maxiter=200))
                if con['fun'](r.x)[0] >= -1e-6 and r.x.sum() < best[0]:
                    e = rk4(f, np.array(x), 0, 0) if False else None
                    best = (float(r.x.sum()), dict(n=n, sign=sg, d=r.x.round(3).tolist()))
        print(q, 'до', n, 'сегм: лучший T/Vf %.3f' % (best[0] / vf), best[1], flush=True)
    out[q] = dict(Vf=vf, T=best[0], ratio=best[0] / vf, sol=best[1])
json.dump(out, open(os.path.join(HERE, 'pend_bruteforce.json'), 'w'), indent=1)
