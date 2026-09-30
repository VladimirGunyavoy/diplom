"""H1 манипулятор 2 звена, динамика 4D: сходимость V стыка-проигрыша по бюджету (эталона нет — самосогласованность; мелкая сетка на aida — потом).
Запуск из v6: python3 tests/check_nd_manip_dyn.py"""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.manip2dyn import flow4
tau = 0.4; Rq = 0.3; Rw = 0.5; WM = 3.0; rho = 0.15
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.max(np.abs(wr(P[..., :2])), -1) < Rq) & (np.max(np.abs(P[..., 2:]), -1) < Rw)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):                                    # затравки на границе окна цели (совет research)
    v = rng.uniform(-1, 1, 4); v = v / np.max(np.abs(v)) * 0.99; seeds.append(v * np.array([Rq, Rq, Rw, Rw]))
S = SysN(flow4, 4, (np.pi, np.pi, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 0, 0), ok=lambda p: np.max(np.abs(p[2:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(8)]
for NB, NF in ((200, 200), (600, 600), (1500, 1500)):
    t0 = time.time(); back = build_back(S, tau, NB, rho); V = []
    for x in Q: V.append(replay_value_fast(S, tau, x, NB, NF, rho, rho, back=back)[0])
    print('NB %d NF %d (%d спор back): V' % (NB, NF, len(back[0])), np.round(V, 2), '(%.0f с)' % (time.time() - t0), flush=True)
