"""С ПРЕПЯТСТВИЯМИ (2 диска). Коридор на адаптивном дереве, манипулятор 3 звена 6D (динамика): T коридора vs V проигрыша (NB, NF). Запуск из v6: python3 tests/check_corridor_manip_dyn.py [NB NF K]"""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.manip3dyn import flow, clearance
TR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NB, NF, K = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (1500, 200, 3)
tau = 0.4; Rq = 0.3; Rw = 0.6; WM = 3.0; rho = 0.15
import os
GG = float(os.environ.get('G', 0)); C3 = np.array([-np.pi / 2 if os.environ.get('DOWN') else 0.0, 0, 0, 0, 0, 0])
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow(P, s, t, dt_max=float(os.environ.get("DT", 0.05)), g=GG)
ing = lambda P: (np.max(np.abs(wr(P[..., :3] - C3[:3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0] - C3[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3] - C3[:3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C3 + v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
OB = [((1.0, -1.5), 0.3), ((-1.0, -1.0), 0.3)]; clear0 = lambda P: clearance(P, OB); clear = lambda P: clear0(P) - 0.02; blk = lambda P: clear0(np.asarray(P)) < 0
S = SysN(fl, 8, (np.pi, np.pi, np.pi, WM, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM, blocked=blk)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(40)]; Q = [x for x in Q if clear0(x) > 0.05][:4]
back = build_back(S, tau, NB, rho)
from src.atlas6.corridor_nd import candidates, refine
x = Q[3]; print('q4', np.round(x, 2), 'miss0', round(float(miss(x)), 2), flush=True)
C, nh = candidates(S, x, back, tau, NF, rho, miss, 8); ne = [c for c in C if c[2]]
print('кандидатов', len(C), 'попаданий', nh, 'непустых', len(ne), 'лучшие miss', [round(c[1], 2) for c in sorted(ne, key=lambda c: c[1])[:5]], flush=True)
for k, (_, _, p) in enumerate(sorted(ne, key=lambda c: c[1])[:3]):
    t0 = time.time(); T, sq, d, ok = refine(fl, x, p, g, tries=5, clear=clear); print(k, 'ok', ok, 'T', round(T, 2), '%.0f с' % (time.time() - t0), flush=True)
