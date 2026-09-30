"""Коридор на адаптивном дереве, дд 3D, 30 запросов: T/ref_window2 mean ≤ 1.01, 0 невалидных. Запуск из spores_2/v6: python3 tests/check_corridor_nd_dd.py [M]"""
import sys, os, json; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
NB, NF = 1200, 50; tau = 0.25; R = .25; Rth = .26; rho = 0.08
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
miss = lambda X: max(np.hypot(X[0], X[1]) - R, 0) + max(abs(dth(X[2])) - Rth, 0)
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
E = json.load(open('../v5chain/reports/research/multiquery_dd_ref.json')); M = int(sys.argv[1]) if len(sys.argv) > 1 else len(E)
rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(len(E))][:M]
back = build_back(S, tau, NB, rho); r = []; bad = 0
for x, e in zip(Q, E):
    b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss)
    if b is None: bad += 1; continue
    xe = np.array(x, float)
    for s, t in zip(b[1], b[2]): xe = fl(xe, s, t)
    if np.any(g(xe) < -1e-6): bad += 1; continue
    r.append(b[0] / e)
r = np.array(r); print('допустимо %d/%d, невалидных %d, T/ref mean %.3f max %.3f' % (len(r), M, bad, r.mean(), r.max()))
assert bad == 0 and r.mean() <= 1.01
print('OK')
