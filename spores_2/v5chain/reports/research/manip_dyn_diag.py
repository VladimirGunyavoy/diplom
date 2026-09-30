"""Диагностика коридора манипулятора 4D (постановка tests/check_corridor_manip_dyn.py worker-6: 5/8 допустимо): для каждого запроса —
V проигрыша, число кандидатов-попаданий/промахов, результат corridor_query. Где рвётся: нет стыка / нет попаданий / SLSQP не сходится.
Запуск из spores_2/v6: python3 ../v5chain/reports/research/manip_dyn_diag.py [NB NF]"""
import sys, time, json, os; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query, candidates, merge
from src.atlas6.manip2dyn import flow4
HERE = os.path.dirname(os.path.abspath(__file__))
NB, NF = [int(a) for a in sys.argv[1:3]] if len(sys.argv) > 2 else (600, 100)
tau = 0.4; Rq = 0.3; Rw = 0.5; WM = 3.0; rho = 0.15
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow4(P, s, t, dt_max=0.05)
ing = lambda P: (np.max(np.abs(wr(P[..., :2])), -1) < Rq) & (np.max(np.abs(P[..., 2:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rw ** 2 - x[2] ** 2, Rw ** 2 - x[3] ** 2])
miss = lambda X: max(np.max(np.abs(wr(X[:2]))) - Rq, 0) + max(np.max(np.abs(X[2:])) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 4); v = v / np.max(np.abs(v)) * 0.99; seeds.append(v * np.array([Rq, Rq, Rw, Rw]))
S = SysN(fl, 4, (np.pi, np.pi, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 0, 0), ok=lambda p: np.max(np.abs(p[2:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(8)]
t0 = time.time(); back = build_back(S, tau, NB, rho); print('обратное дерево', len(back[0]), 'спор, %.0f с' % (time.time() - t0), flush=True)
rows = []
for q, x in enumerate(Q):
    t0 = time.time(); V = replay_value_fast(S, tau, x, NB, NF, rho, rho, back=back)[0]
    C, nhit = candidates(S, x, back, tau, NF, rho, miss)
    tops_h = len({tuple(merge(p)[0]) for _, _, p in C[:nhit] if p}); tops_m = len({tuple(merge(p)[0]) for _, _, p in C[nhit:] if p})
    b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=3)
    row = dict(q=q, x=np.round(x, 2).tolist(), V=V, n_hit=nhit, n_miss=len(C) - nhit, top_hit=tops_h, top_miss=tops_m,
               best_miss=float(C[nhit][1]) if len(C) > nhit else None, T=None if b is None else b[0], seq=None if b is None else list(map(int, b[1])), sec=time.time() - t0)
    rows.append(row); print(row, flush=True)
json.dump(rows, open(os.path.join(HERE, 'manip_dyn_diag_%d_%d.json' % (NB, NF)), 'w'), indent=1)
