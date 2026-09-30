"""Многозапросность (очередь research №2): дифдрайв 3D ромб, общее обратное дерево NB на M запросов, прямое NF на каждый.
Сетка (NB, NF): покрытие, V/ref_window2, время постройки и на запрос. Итог: сколько спор на M запросов (NB + M·NF).
Запуск из spores_2/v6: python3 ../v5chain/reports/research/multiquery_dd.py [M]"""
import sys, os, json, time, importlib.util; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('w', os.path.join(HERE, 'dd_window_ref.py')); w = importlib.util.module_from_spec(sp); sp.loader.exec_module(w)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 30; tau = 0.25; R = .25; Rth = .26; rho = 0.08
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(M)]
cache = os.path.join(HERE, 'multiquery_dd_ref.json')
E = json.load(open(cache)) if os.path.exists(cache) else []
if len(E) < M:
    t0 = time.time(); E = [float(w.ref_window2(p)) for p in Q]; json.dump(E, open(cache, 'w')); print('эталон %d запросов %.0f с' % (M, time.time() - t0), flush=True)
E = np.array(E[:M]); out = []
for NB in (300, 1200, 4800):
    t0 = time.time(); back = build_back(S, tau, NB, rho); tb = time.time() - t0
    for NF in (1, 10, 50, 200, 600):
        t0 = time.time(); V = np.array([replay_value(S, tau, x, NB, NF, rho, rho, back=back)[0] for x in Q]); tq = (time.time() - t0) / M
        ok = np.isfinite(V); r = V[ok] / E[ok]
        row = dict(NB=len(back[0]), NF=NF, cov=int(ok.sum()), mean=float(r.mean()) if ok.any() else None, max=float(r.max()) if ok.any() else None,
                   min=float(r.min()) if ok.any() else None, t_back=tb, t_query=tq, spores_M=len(back[0]) + M * NF)
        out.append(row); print(row, flush=True)
json.dump(dict(M=M, tau=tau, rho=rho, rows=out), open(os.path.join(HERE, 'multiquery_dd.json'), 'w'), indent=1)
