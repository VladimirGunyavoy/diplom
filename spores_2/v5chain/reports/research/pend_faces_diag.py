"""research hub-research-7: диагностика достижимости маятника на гранях (v7 faces.py). Гипотеза пользователя: недостижимых стартов нет физически."""
import sys, time, json; sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research/snap')
import numpy as np
from src.cells7.systems import Sys2
from src.cells7.faces import Faces, lines_graded
def pend_top(umax=.3, wmax=4.0):
    f = lambda x, u: np.stack([x[..., 1], np.sin(x[..., 0]) + u], -1)
    A = lambda x, u: np.array([[0.0, 1.0], [np.cos(x[0]), 0.0]])
    return Sys2('pend_top', (np.pi, wmax), f, A, umax, per=(2.0, 0.0), box=((-1, 1), (-1, 1)))
rng = np.random.default_rng(0)
Qphys = np.stack([rng.uniform(-1, 1, 300) * np.pi, rng.uniform(-.8, .8, 300) * 4.0], 1)      # те же старты, физ. (φ, ω)
d0 = float(sys.argv[1]) if len(sys.argv) > 1 else .05; res = []
for wmax in [float(a) for a in sys.argv[2:]] or (4.0, 6.0):
    S = pend_top(wmax=wmax); Q = Qphys / np.array([np.pi, wmax]); ln = lines_graded(.02, 1.0, 400, d0=d0)
    t0 = time.time(); F = Faces(S, ln, ln, m=4, periodic_x=True, tmax=40.0); V = F.solve(); T, _ = F.rollout(Q); fin = np.isfinite(T)
    E = Qphys[:, 1]**2 / 2 + np.cos(Qphys[:, 0])
    r = dict(wmax=wmax, d0=d0, probes=len(F.P), finV=round(float(np.isfinite(V).mean()), 3), reach=round(float(fin.mean()), 3), sec=round(time.time() - t0),
             fail_phi=np.round(np.percentile(Qphys[~fin, 0], [10, 50, 90]), 2).tolist() if (~fin).any() else [],
             fail_w=np.round(np.percentile(Qphys[~fin, 1], [10, 50, 90]), 2).tolist() if (~fin).any() else [],
             fail_E=np.round(np.percentile(E[~fin], [10, 50, 90]), 2).tolist() if (~fin).any() else [], ok_E=np.round(np.percentile(E[fin], [10, 50, 90]), 2).tolist())
    print(json.dumps(r, ensure_ascii=False), flush=True); res.append(r)
    np.save(f'/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research/pend_faces_diag_w{wmax:g}_d{d0:g}.npy', np.c_[Qphys, T])
