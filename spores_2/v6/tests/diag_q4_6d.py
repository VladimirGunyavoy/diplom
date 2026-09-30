"""Коридор на адаптивном дереве, манипулятор 3 звена 6D (динамика): T коридора vs V проигрыша (NB, NF). Запуск из v6: python3 tests/check_corridor_manip_dyn.py [NB NF K]"""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.manip3dyn import flow
NB, NF, K = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (1500, 200, 3)
tau = 0.4; Rq = 0.3; Rw = 0.6; WM = 3.0; rho = 0.15
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow(P, s, t, dt_max=0.05)
ing = lambda P: (np.max(np.abs(wr(P[..., :3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
S = SysN(fl, 8, (np.pi, np.pi, np.pi, WM, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(4)]
back = build_back(S, tau, NB, rho)
from src.atlas6.corridor_nd import candidates, refine, merge
x = Q[3]; C, nh = candidates(S, x, back, tau, NF, rho, miss, 8); print('кандидатов', len(C), 'попаданий', nh, 'длины', [len(c[2]) for c in C[:8]], 'miss', [round(c[1], 2) for c in C[:8]], flush=True)
NE = sorted([c for c in C if c[2]], key=lambda c: c[1])[:5]; print('непустых', sum(1 for c in C if c[2]), 'лучшие miss', [round(c[1], 2) for c in NE], 'длины', [len(c[2]) for c in NE], flush=True)
for k, (_, _, p) in enumerate(NE[:4]):
    t0 = time.time(); T, sq, d, ok = refine(fl, x, p, g, tries=8); print(k, 'ok', ok, 'T', round(T, 2), '%.0f с' % (time.time() - t0), flush=True)
