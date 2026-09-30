"""Манипулятор 4D (динамика, 4 слоя-угла моментов): независимый эталон перебором топологий ≤ NMAX сегментов × starts стартов SLSQP
(min Σdt, конец в окне). Постановка и запросы — tests/check_corridor_manip_dyn.py (worker-6). Кто прав на q1/q3/q5 (коридор дерева не нашёл).
Запуск из spores_2/v6: [ONLY=1,3] python3 ../v5chain/reports/research/manip_bruteforce.py [NMAX starts]"""
import sys, os, json, time, itertools; sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize
from src.atlas6.manip2dyn import flow4
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX, ST = [int(a) for a in sys.argv[1:3]] if len(sys.argv) > 2 else (3, 3)
Rq = 0.3; Rw = 0.5
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rw ** 2 - x[2] ** 2, Rw ** 2 - x[3] ** 2])
rng = np.random.default_rng(0)
for _ in range(32): v = rng.uniform(-1, 1, 4)           # тот же поток случайных чисел, что у worker (затравки), затем запросы
Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(8)]
def end(x, seq, d):
    X = np.array(x, float)
    for s, t in zip(seq, d): X = flow4(X, s, t, dt_max=0.05)
    return X
ONLY = [int(a) for a in os.environ.get('ONLY', '').split(',') if a]; out = {}
for q, x in enumerate(Q):
    if ONLY and q not in ONLY: continue
    t0 = time.time(); r2 = np.random.default_rng(q); best = (np.inf, None)
    for n in range(1, NMAX + 1):
        for seq in itertools.product(range(4), repeat=n):
            if any(seq[i] == seq[i + 1] for i in range(n - 1)): continue
            con = {'type': 'ineq', 'fun': lambda d, seq=seq: g(end(x, seq, d))}
            for _ in range(ST):
                d0 = r2.uniform(0.1, 1.5, n)
                r = minimize(lambda d: d.sum(), d0, jac=lambda d: np.ones_like(d), bounds=[(0, 4.0)] * n, constraints=[con], method='SLSQP', options=dict(maxiter=200))
                if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), dict(seq=list(seq), d=r.x.round(3).tolist()))
        print(q, 'до', n, 'сегм: T %.3f' % best[0], best[1], '%.0f с' % (time.time() - t0), flush=True)
    xe = end(x, best[1]['seq'], best[1]['d']) if best[1] else None
    out[q] = dict(x=x.tolist(), T=best[0], sol=best[1], end_ok=None if xe is None else bool(np.all(g(xe) > -1e-3)))
json.dump(out, open(os.path.join(HERE, 'manip_bruteforce%s.json' % ('_' + '_'.join(map(str, ONLY)) if ONLY else '')), 'w'), indent=1)
