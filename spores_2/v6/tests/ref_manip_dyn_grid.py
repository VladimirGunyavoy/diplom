"""Эталон-грубая сетка для манипулятора 4D (полулагранжева V-итерация, τ=.4, 4 слоя): сетка q 48², w 31² по [-3,3]; запросы как в check_nd_manip_dyn.
Оценка ПРИБЛИЖЁННАЯ (линейная интерполяция → завышение у цели, окно по узлам). Запуск из v6: python3 tests/ref_manip_dyn_grid.py"""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from scipy.ndimage import map_coordinates
from src.atlas6.manip2dyn import flow4
tau = 0.4; Rq = 0.3; Rw = 0.5; WM = 3.0; nq = int(sys.argv[1]) if len(sys.argv) > 1 else 48; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 31
hq = 2 * np.pi / nq; hw = 2 * WM / (nw - 1)
gq = -np.pi + hq * np.arange(nq); gw = -WM + hw * np.arange(nw)
X = np.stack(np.meshgrid(gq, gq, gw, gw, indexing='ij'), -1).reshape(-1, 4)
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
goal = ((np.max(np.abs(wr(X[:, :2])), -1) < Rq) & (np.max(np.abs(X[:, 2:]), -1) < Rw)).reshape(nq, nq, nw, nw)
K = 60.0; C = []; OK = []
for s in range(4):
    E = flow4(X, s, tau, dt_max=0.05); OK.append(np.max(np.abs(E[:, 2:]), -1) <= WM)
    c = np.stack([(E[:, 0] + np.pi) / hq % nq, (E[:, 1] + np.pi) / hq % nq, np.clip((E[:, 2] + WM) / hw, 0, nw - 1), np.clip((E[:, 3] + WM) / hw, 0, nw - 1)]); C.append(c)
V = np.full(goal.shape, K); V[goal] = 0.0; t0 = time.time()
for it in range(400):
    Vn = np.full(V.size, K)
    for s in range(4):
        v = map_coordinates(V, C[s], order=1, mode='grid-wrap') + tau; v[~OK[s]] = K; Vn = np.minimum(Vn, v)
    Vn = np.minimum(Vn, K).reshape(V.shape); Vn[goal] = 0.0; d = np.abs(Vn - V).max(); V = Vn
    if it % 10 == 0: print(it, d, '%.0f с' % (time.time() - t0), flush=True)
    if d < 1e-6: break
rng = np.random.default_rng(0); [rng.uniform(-1, 1, 4) for _ in range(32)]
Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(8)]
c = lambda x: [[(x[0] + np.pi) / hq % nq], [(x[1] + np.pi) / hq % nq], [(x[2] + WM) / hw], [(x[3] + WM) / hw]]
print('V_grid', np.round([map_coordinates(V, c(x), order=1, mode='grid-wrap')[0] for x in Q], 2))
