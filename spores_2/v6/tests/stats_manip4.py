"""(б) hub-worker-8: 4 звена 8D, динамика, g=.3, цель вниз (q1=-π/2). env NQ, NAME, G. Запуск из v6: python3 tests/stats_manip4.py NB NF K [tries]. JSON — reports/stats_<NAME>.json"""
import sys, os, time, json; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, back_heuristic
from src.atlas6.corridor_nd import corridor_batch
from src.atlas6.manip3dyn import flow
n = 4; NB, NF, K = [int(a) for a in sys.argv[1:4]]; TR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
tau = 0.4; Rq = 0.3; Rw = 0.6; WM = 3.0; rho = 0.15; GG = float(os.environ.get('G', .3))
C = np.zeros(2 * n); C[0] = -np.pi / 2
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow(P, s, t, dt_max=float(os.environ.get('DT', 0.02)), n=n, g=GG)
ing = lambda P: (np.max(np.abs(wr(P[..., :n] - C[:n])), -1) < Rq) & (np.max(np.abs(P[..., n:]), -1) < Rw)
g = lambda x: np.concatenate([Rq ** 2 - wr(x[:n] - C[:n]) ** 2, Rw ** 2 - x[n:] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :n] - C[:n])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., n:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []; R8 = np.array([Rq] * n + [Rw] * n)
for _ in range(32):
    v = rng.uniform(-1, 1, 2 * n); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C + v * R8)
S = SysN(fl, 2 ** n, (np.pi,) * n + (WM,) * n, ing, seeds, per=(2 * np.pi,) * n + (0,) * n, ok=lambda p: np.max(np.abs(p[n:])) <= WM)
NQ = int(os.environ.get('NQ', 4)); Q = [np.array([*rng.uniform(-np.pi, np.pi, n), *rng.uniform(-1, 1, n)]) for _ in range(NQ)]
t0 = time.time(); back = build_back(S, tau, NB, rho); tb = time.time() - t0
t0 = time.time(); Rr = corridor_batch(S, fl, Q, back, tau, NF, rho, g, miss, K=K, tries=TR, hfun=back_heuristic(S, back), kn=8); tt = time.time() - t0
out = []
for x, b in zip(Q, Rr):
    ok = False
    if b is not None:
        xe = np.array(x, float)
        for s, d in zip(b[1], b[2]): xe = flow(xe, s, d, dt_max=0.005, n=n, g=GG)
        ok = bool(np.all(g(xe) > -1e-4))
    out.append(dict(T=None if b is None else float(b[0]), ok=ok))
Ts = [o['T'] for o in out if o['ok']]
print('NAME', os.environ.get('NAME'), 'NB', NB, 'NF', NF, 'валидно %d/%d' % (len(Ts), NQ), 'T', [round(t, 2) for t in Ts], 'back %.0f с, серия %.0f с' % (tb, tt), flush=True)
json.dump(dict(NB=NB, NF=NF, K=K, tback=tb, tsec=tt, res=out), open('reports/stats_%s.json' % os.environ.get('NAME', 'x'), 'w'))
