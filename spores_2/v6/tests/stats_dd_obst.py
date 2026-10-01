"""СТАТИСТИКА hub-worker-8: 40 запросов (corridor_batch), dd 3D + 2 диска; V сетки dd_atlas h=.125 как опорная верхняя оценка. env NQ, NAME. Коридор на адаптивном дереве, dd 3D с 2 дисками: T коридора (SLSQP с зазором) vs V проигрыша vs сетка dd_atlas; путь пересчитан мелким dt на столкновения. Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS, solve_dd, V_at
tau = 0.25; R = .25; Rth = .26; rho = 0.08; NB, NF = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1200, 100)
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
miss = lambda X: np.maximum(np.hypot(X[..., 0], X[..., 1]) - R, 0) + np.maximum(np.abs(dth(X[..., 2])) - Rth, 0)
OB = [(0.9, 0.9, 0.5), (-0.9, -0.5, 0.4)]
clear0 = lambda P: np.min([np.hypot(P[..., 0] - o[0], P[..., 1] - o[1]) - o[2] for o in OB], axis=0)
clear = lambda P: clear0(P) - 0.02      # запас на межточечный зазор
blk = lambda P: clear0(np.asarray(P)) < 0
So = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3, blocked=blk)
A = solve_dd(h=0.125, tau=tau, R_goal=R, Rth=Rth, obstacles=OB)
import os, json
NQ = int(os.environ.get('NQ', 40)); rng = np.random.default_rng(3); Q = []
while len(Q) < NQ:
    x = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)])
    if not blk(x): Q.append(x)
back = build_back(So, tau, NB, rho)
from src.atlas6.adaptive_nd import back_heuristic
from src.atlas6.corridor_nd import corridor_batch
t0 = time.time(); R_ = corridor_batch(So, fl, Q, back, tau, NF, rho, g, miss, K=3, tries=3, clear=clear, hfun=back_heuristic(So, back)); tt = time.time() - t0
out = []
for x, b in zip(Q, R_):
    ok = False; col = None
    if b is not None:
        X = np.array(x, float); col = 0
        for s, d in zip(b[1], b[2]):
            m = max(1, int(np.ceil(d / 0.01)))
            for _ in range(m): X = flow(X, LAYERS[s], d / m); col += int(blk(X))
        ok = col == 0 and bool(np.all(g(X) > -1e-4))
    out.append(dict(T=None if b is None else float(b[0]), ok=bool(ok), col=col, Vg=float(V_at(A, x))))
r = [o['T'] / o['Vg'] for o in out if o['ok'] and 0 < o['Vg'] < 50]
print('NAME', os.environ.get('NAME'), 'NB', NB, 'NF', NF, 'валидно %d/%d' % (sum(o['ok'] for o in out), NQ), 'столкновений в решённых %d' % sum((o['col'] or 0) > 0 for o in out), 'T/V_сетки mean %.3f med %.3f max %.3f (n=%d)' % (np.mean(r), np.median(r), np.max(r), len(r)), 'серия %.0f с' % tt, flush=True)
json.dump(dict(NB=NB, NF=NF, tsec=tt, res=out), open('reports/stats_%s.json' % os.environ.get('NAME', 'x'), 'w'))
