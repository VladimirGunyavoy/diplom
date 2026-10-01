"""(г) hub-worker-8: адаптив vs равномерная сетка при РАВНОМ числе узлов, манипулятор 2 звена 4D без g, 4 слоя, 8 запросов (как ref_manip_dyn_grid.py).
V_adaptive = replay_value_fast(NB=N, NF=N/6), V_grid = полулагранжева V-итерация на nq²·nw² узлах; ref = T коридора NB2400 NF400 (= эталон research). Запуск из v6: python3 tests/adaptive_vs_grid_4d.py"""
import sys, time, json; sys.path.insert(0, '.')
import numpy as np
from scipy.ndimage import map_coordinates
from src.atlas6.adaptive_nd import SysN, build_back, replay_value_fast, back_heuristic
from src.atlas6.corridor_nd import corridor_batch
from src.atlas6.manip2dyn import flow4
tau = 0.4; Rq = 0.3; Rw = 0.5; WM = 3.0; rho = 0.15
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow4(P, s, t, dt_max=0.05)
ing = lambda P: (np.max(np.abs(wr(P[..., :2])), -1) < Rq) & (np.max(np.abs(P[..., 2:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rw ** 2 - x[2] ** 2, Rw ** 2 - x[3] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :2])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 2:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 4); v = v / np.max(np.abs(v)) * 0.99; seeds.append(v * np.array([Rq, Rq, Rw, Rw]))
S = SysN(fl, 4, (np.pi, np.pi, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 0, 0), ok=lambda p: np.max(np.abs(p[2:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(8)]
out = {}
back = build_back(S, tau, 2400, rho); R = corridor_batch(S, fl, Q, back, tau, 400, rho, g, miss, K=3, tries=3, hfun=back_heuristic(S, back), kn=8)
ref = [None if b is None else float(b[0]) for b in R]; out['ref'] = ref; print('ref T', np.round([r if r else -1 for r in ref], 2), flush=True)
for NB in (600, 2400, 9000, 25000):
    t0 = time.time(); back = build_back(S, tau, NB, rho)
    V = [replay_value_fast(S, tau, x, NB, max(NB // 6, 100), rho, rho, back=back)[0] for x in Q]
    out['adapt_%d' % NB] = [float(v) for v in V]; print('адаптив N=%d:' % NB, np.round(V, 2), '%.0f с' % (time.time() - t0), flush=True)
def grid(nq, nw):
    hq = 2 * np.pi / nq; hw = 2 * WM / (nw - 1); gq = -np.pi + hq * np.arange(nq); gw = -WM + hw * np.arange(nw)
    X = np.stack(np.meshgrid(gq, gq, gw, gw, indexing='ij'), -1).reshape(-1, 4)
    goal = ((np.max(np.abs(wr(X[:, :2])), -1) < Rq) & (np.max(np.abs(X[:, 2:]), -1) < Rw)).reshape(nq, nq, nw, nw)
    K = 60.0; C = []; OK = []
    for s in range(4):
        E = flow4(X, s, tau, dt_max=0.05); OK.append(np.max(np.abs(E[:, 2:]), -1) <= WM)
        C.append(np.stack([(E[:, 0] + np.pi) / hq % nq, (E[:, 1] + np.pi) / hq % nq, np.clip((E[:, 2] + WM) / hw, 0, nw - 1), np.clip((E[:, 3] + WM) / hw, 0, nw - 1)]))
    V = np.full(goal.shape, K); V[goal] = 0.0
    for it in range(400):
        Vn = np.full(V.size, K)
        for s in range(4):
            v = map_coordinates(V, C[s], order=1, mode='grid-wrap') + tau; v[~OK[s]] = K; Vn = np.minimum(Vn, v)
        Vn = np.minimum(Vn, K).reshape(V.shape); Vn[goal] = 0.0; d = np.abs(Vn - V).max(); V = Vn
        if d < 1e-6: break
    c = lambda x: [[(x[0] + np.pi) / hq % nq], [(x[1] + np.pi) / hq % nq], [(x[2] + WM) / hw], [(x[3] + WM) / hw]]
    return [float(map_coordinates(V, c(x), order=1, mode='grid-wrap')[0]) for x in Q]
for nq, nw in ((8, 6), (12, 8), (16, 10), (24, 16)):
    t0 = time.time(); V = grid(nq, nw); out['grid_%d' % (nq * nq * nw * nw)] = V; print('сетка N=%d:' % (nq * nq * nw * nw), np.round(V, 2), '%.0f с' % (time.time() - t0), flush=True)
json.dump(out, open('reports/adaptive_vs_grid_4d.json', 'w'))
