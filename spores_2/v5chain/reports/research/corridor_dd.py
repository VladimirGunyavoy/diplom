"""Коридор на пути адаптивного дерева, дифдрайв 3D ромб: V проигрыша → T коридора (SLSQP) против ref_window2 (кэш multiquery_dd_ref.json).
Запуск из spores_2/v6: python3 ../v5chain/reports/research/corridor_dd.py [NB NF]"""
import sys, os, json, time, importlib.util; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from corridor_nd import refine
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1200; NF = int(sys.argv[2]) if len(sys.argv) > 2 else 200
tau = 0.25; R = .25; Rth = .26; rho = 0.08
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
E = json.load(open(os.path.join(HERE, 'multiquery_dd_ref.json'))); M = len(E)
rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(M)]
back = build_back(S, tau, NB, rho); rows = []
for x, e in zip(Q, E):
    t0 = time.time(); V, _, _, path = replay_value(S, tau, x, NB, NF, rho, rho, back=back)
    if not np.isfinite(V): rows.append(dict(V=None)); continue
    T, seq, d, ok = refine(fl, x, path, g)
    rows.append(dict(V=V / e, T=T / e, ok=ok, nseg=len(seq), sec=time.time() - t0)); print(rows[-1], flush=True)
v = np.array([r['V'] for r in rows if r['V']]); t = np.array([r['T'] for r in rows if r['V']])
print('NB %d NF %d: найдено %d/%d; V/ref mean %.3f max %.3f; коридор T/ref mean %.3f min %.3f max %.3f' % (NB, NF, len(v), M, v.mean(), v.max(), t.mean(), t.min(), t.max()))
json.dump(dict(NB=NB, NF=NF, rows=rows), open(os.path.join(HERE, 'corridor_dd_%d_%d.json' % (NB, NF)), 'w'), indent=1)
