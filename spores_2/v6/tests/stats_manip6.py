"""СТАТИСТИКА (hub-worker-8): NQ случайных запросов параллельно (corridor_batch), 3 звена 6D. env: G, DOWN/UP, NQ, OBST=1, NAME. Запуск из v6: python3 tests/stats_manip6.py NB NF K [tries]; JSON — reports/stats_<NAME>.json. Коридор на адаптивном дереве, манипулятор 3 звена 6D (динамика): T коридора vs V проигрыша (NB, NF). Запуск из v6: python3 tests/check_corridor_manip_dyn.py [NB NF K]"""
import sys, os, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.manip3dyn import flow, clearance
TR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NB, NF, K = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (1500, 200, 3)
tau = 0.4; Rq = 0.3; Rw = 0.6; WM = float(os.environ.get('WM', 3.0)); rho = 0.15
import os
GG = float(os.environ.get('G', 0)); C3 = np.array([-np.pi / 2 if os.environ.get('DOWN') else (np.pi / 2 if os.environ.get('UP') else 0.0), 0, 0, 0, 0, 0])
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow(P, s, t, dt_max=float(os.environ.get("DT", 0.05)), g=GG)
ing = lambda P: (np.max(np.abs(wr(P[..., :3] - C3[:3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0] - C3[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3] - C3[:3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C3 + v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
OB = [((1.0, -1.5), 0.3), ((-1.0, -1.0), 0.3)] if os.environ.get('OBST') else [((9.0, 9.0), 0.1)]; clear0 = lambda P: clearance(P, OB); clear = lambda P: clear0(P) - 0.02; blk = lambda P: clear0(np.asarray(P)) < 0
S = SysN(fl, 8, (np.pi, np.pi, np.pi, WM, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM, blocked=blk)
NQ = int(os.environ.get('NQ', 30)); Q = []
while len(Q) < NQ:
    x = np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)])
    if clear0(x) > 0.05: Q.append(x)
back = build_back(S, tau, NB, rho)
from src.atlas6.adaptive_nd import back_heuristic
from src.atlas6.corridor_nd import corridor_batch
HF = back_heuristic(S, back, wh=(float(os.environ['WH']) if os.environ.get('WH') else None)) if os.environ.get('WH') != '0' else None
t0 = time.time(); R = corridor_batch(S, fl, Q, back, tau, NF, rho, g, miss, K=K, tries=TR, hfun=HF, clear=clear, kn=int(os.environ.get('KN', 8)), fs=int(os.environ.get('FS', 0))); tt = time.time() - t0
out = []
for x, b in zip(Q, R):
    ok = False; mc = None
    if b is not None:
        xe = np.array(x, float); mc = clear0(xe)
        for s, d in zip(b[1], b[2]):
            for _ in range(8): xe = flow(xe, s, d / 8, dt_max=0.005, g=GG); mc = min(mc, clear0(xe))
        ok = bool(np.all(g(xe) > -1e-4)) and bool(mc >= 0)
    out.append(dict(T=None if b is None else float(b[0]), ok=ok, minclear=None if mc is None else float(mc), seq=None if b is None else [int(s) for s in b[1]], dts=None if b is None else [float(d) for d in b[2]]))
Ts = [o['T'] for o in out if o['ok']]
print('NAME', os.environ.get('NAME'), 'NB', NB, 'NF', NF, 'решено валидно %d/%d' % (len(Ts), NQ), 'медиана T %.2f' % (np.median(Ts) if Ts else -1), 'время серии %.0f с' % tt, flush=True)
import json; json.dump(dict(NB=NB, NF=NF, K=K, tries=TR, tsec=tt, res=out), open('reports/stats_%s.json' % os.environ.get('NAME', 'x'), 'w'))
