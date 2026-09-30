"""Коридор на адаптивном дереве, 2 звена с ГРАВИТАЦИЕЙ 4D (вертикально, g=.3, |τ|≤1: слабые моменты, раскачка; цель — вертикально вверх, неустойчивая): T коридора vs V проигрыша (NB, NF). Запуск из v6: python3 tests/check_corridor_manip_dyn.py [NB NF K]"""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.manip3dyn import flow as _fl
NB, NF, K = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (600, 300, 3)
tau = 0.4; Rq = 0.3; Rw = 0.5; WM = 3.0; rho = 0.15
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
GG = 0.3; fl = lambda P, s, t: _fl(P, s, t, dt_max=0.05, n=2, g=GG)
UP = np.array([np.pi / 2, 0.0, 0.0, 0.0])
ing = lambda P: (np.max(np.abs(wr(P[..., :2] - UP[:2])), -1) < Rq) & (np.max(np.abs(P[..., 2:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0] - UP[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rw ** 2 - x[2] ** 2, Rw ** 2 - x[3] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :2] - UP[:2])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 2:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 4); v = v / np.max(np.abs(v)) * 0.99; seeds.append(UP + v * np.array([Rq, Rq, Rw, Rw]))
S = SysN(fl, 4, (np.pi, np.pi, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 0, 0), ok=lambda p: np.max(np.abs(p[2:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(8)]
back = build_back(S, tau, NB, rho)
for x in Q:
    t0 = time.time(); V = replay_value_fast(S, tau, x, NB, NF, rho, rho, back=back)[0]; b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=K)
    xe = None if b is None else np.array(x, float)
    if b is not None:
        for s, d in zip(b[1], b[2]): xe = _fl(xe, s, d, dt_max=0.005, n=2, g=GG)
    print('V %.2f T_corr %s конец в окне(мелкий rk4): %s (%.0f с)' % (V, None if b is None else round(b[0], 2), None if b is None else bool(np.all(g(xe) > -1e-4)), time.time() - t0), flush=True)
