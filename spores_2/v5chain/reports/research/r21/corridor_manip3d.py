"""research-21: 3D двузвенный манипулятор (manip3d.py) — коридор v6 (адаптивное дерево + SLSQP), как check_corridor_manip6.py для 3 зв.
Цель — покой в q* = 0 (рука горизонтально вытянута, рыскание 0): окно |q| < Rq, |w| < Rw. Запуск из v6: python3 corridor_manip3d.py NB NF K [NQ]"""
import sys, time, os; sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
from manip3d import flow, US
NB, NF, K = [int(a) for a in sys.argv[1:4]]; NQ = int(sys.argv[4]) if len(sys.argv) > 4 else 8; TR = 3
tau = 0.4; Rq = 0.3; Rw = 0.6; WM = 3.0; rho = 0.15; C3 = np.zeros(6)
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow(P, s, t, dt_max=float(os.environ.get('DT', .05)))
ing = lambda P: (np.max(np.abs(wr(P[..., :3] - C3[:3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C3 + v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
S = SysN(fl, 8, (np.pi, np.pi, np.pi, WM, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(NQ)]
t0 = time.time(); back = build_back(S, tau, NB, rho); tb = time.time() - t0; print('обратное дерево NB %d: %.0f с' % (NB, tb), flush=True)
res = []
from tqdm import tqdm
for i, x in tqdm(list(enumerate(Q)), desc='запросы коридора', mininterval=float(os.environ.get('TQDM_MI', 10))):
    t0 = time.time(); V = replay_value_fast(S, tau, x, NB, NF, rho, rho, back=back)[0]; b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=K, tries=TR, kn=8)
    xe = None if b is None else np.array(x, float); wmax = 0.
    if b is not None:
        for s, d in zip(b[1], b[2]):
            for _ in range(20): xe = flow(xe, s, d / 20, dt_max=.005); wmax = max(wmax, np.abs(xe[3:]).max())
    ok = b is not None and bool(np.all(g(xe) > -1e-4)); dt = time.time() - t0
    res.append(dict(q=i, V=round(float(V), 3), T=None if b is None else round(float(b[0]), 3), ok=ok, wmax=round(wmax, 2), sec=round(dt, 1), arcs=None if b is None else len(b[1])))
    print(res[-1], flush=True)
import json; print(json.dumps(dict(NB=NB, NF=NF, K=K, t_back=round(tb, 1), ok=sum(r['ok'] for r in res), n=len(res), res=res)), flush=True)
