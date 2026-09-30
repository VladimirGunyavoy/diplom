"""dd 3D с дисками: адаптивный replay_value_fast против сеточного dd_atlas.solve_dd (h=.125, τ=.25, окно R=.25 Rθ=.26). Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS, solve_dd, V_at
tau = 0.25; R = .25; Rth = .26; rho = 0.08; NB, NF = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1200, 600)
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
OB = [(0.9, 0.9, 0.5), (-0.9, -0.5, 0.4)]; blk = lambda P: np.any([np.hypot(P[..., 0] - o[0], P[..., 1] - o[1]) < o[2] for o in OB], axis=0)
So = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3, blocked=blk)
t0 = time.time(); A = solve_dd(h=0.125, tau=tau, R_goal=R, Rth=Rth, obstacles=OB); print('сетка %.0f с, итераций %d' % (time.time() - t0, A['iters']), flush=True)
rng = np.random.default_rng(3); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(12)]; Q = [x for x in Q if not blk(x)][:10]
back = build_back(So, tau, NB, rho); r = []
for x in Q:
    Vg = V_at(A, x); Va = replay_value_fast(So, tau, x, NB, NF, rho, rho, back=back)[0]; r.append(Va / Vg); print('V сетка %.2f адаптив %.2f отношение %.3f' % (Vg, Va, r[-1]), flush=True)
r = np.array(r); f = np.isfinite(r); print('покрытие %d/%d, отношение mean %.3f min %.3f max %.3f' % (f.sum(), len(r), r[f].mean(), r[f].min(), r[f].max()))
