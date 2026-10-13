"""CA-MAZE режим (б): K=20 целей (центры случайных клеток), PRM* для дд: карта (узлы + рёбра) строится ОДИН раз в мировых координатах, на цель — Dijkstra от узла цели + 50 запросов (те же старты rng(1), отброшены лежащие в цели ближе 1.5);
T_ref на цель — A*+натяжение (maze.ref_T). python prm_maze_goals.py [N=3000] [K=20] → data/maze_prm_goals_N.json"""
import os, sys, time, json, heapq, numpy as np
from scipy.spatial import cKDTree
import common, maze
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; K = int(sys.argv[2]) if len(sys.argv) > 2 else 20; MARG = .02; n = maze.N_CELLS
M0 = maze.Maze(n, 0); rng = np.random.default_rng(0); gfw = lambda p: M0.gfree(np.asarray(p) - M0.G0)       # мир [0,n]²: gfree(y) для y = мир − G0
def los(a, b):
    t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / .01)))[:, None]; return (gfw(a + t * (b - a)) >= MARG - 1e-9).all()
t0 = time.perf_counter(); P = []
while len(P) < N:
    p = rng.uniform(0, n, 2)
    if gfw(p[None])[0] >= MARG: P.append(p)
P = np.array(P); tree = cKDTree(P); adj = [[] for _ in range(N)]; r = 2.0 * np.sqrt(np.log(N) / N) * n * 1.6
for i in range(N):
    for j in tree.query_ball_point(P[i], r):
        if j > i and los(P[i], P[j]): w = float(np.linalg.norm(P[i] - P[j])); adj[i].append((j, w)); adj[j].append((i, w))
t_map = time.perf_counter() - t0; print('карта', round(t_map), 'с', flush=True)
extra_w = {}
def dijk(src, extra):
    """Dijkstra от узла src = N (цель); extra — рёбра цели (узел, вес); остальной граф — общий"""
    d = np.full(N + 1, np.inf); par = -np.ones(N + 1, int); d[N] = 0.; pq = [(0., N)]
    while pq:
        c, u = heapq.heappop(pq)
        if c > d[u]: continue
        for v, w in (extra if u == N else adj[u] + ([(N, extra_w[u])] if u in extra_w else [])):
            if c + w < d[v]: d[v] = c + w; par[v] = u; heapq.heappush(pq, (d[v], v))
    return d, par
def execute(h0, Q):
    seg = np.diff(Q, axis=0); L = np.linalg.norm(seg, axis=1); hd = np.arctan2(seg[:, 1], seg[:, 0]); h = np.r_[h0, hd, 0.]; us = []; dts = []
    for k in range(len(seg)):
        da = float(np.angle(np.exp(1j * (h[k + 1] - h[k]))))
        if abs(da) > 1e-9: us.append([0., np.sign(da)]); dts.append(abs(da))
        us.append([1., 0.]); dts.append(L[k])
    da = float(np.angle(np.exp(1j * (h[-1] - h[-2]))))
    if abs(da) > 1e-9: us.append([0., np.sign(da)]); dts.append(abs(da))
    return us, dts
gl = [(int(a), int(b)) for a, b in rng.integers(0, n, (K, 2))]; res = []; allT = []
for g in gl:
    e = maze.MazeDD(goal=g); G = e.M.G0; t1 = time.perf_counter()
    near = [j for j in tree.query(G, 30)[1] if los(G, P[j])]; extra = [(int(j), float(np.linalg.norm(G - P[j]))) for j in near]; extra_w = {j: w for j, w in extra}
    d, par = dijk(N, extra); t_goal = time.perf_counter() - t1; Ts = []; tq = []; Pn = np.vstack([P, G])
    for y0 in e.X0:
        t2 = time.perf_counter(); a = y0[:2] + G; cand = [j for j in tree.query(a, 30)[1] if np.isfinite(d[j]) and los(a, P[j])]
        if not cand: Ts.append(None); tq.append(time.perf_counter() - t2); continue
        j = min(cand, key=lambda j: np.linalg.norm(a - P[j]) + d[j]); path = [a]
        while j != -1: path.append(Pn[j]); j = par[j]
        Q = np.array(path) - G; us, dts = execute(y0[2], Q); tq.append(time.perf_counter() - t2); Tv, why = common.verify(e, y0, us, dts); Ts.append(Tv)
    s = common.summarize(e, Ts); s.update(goal=g, t_goal=t_goal, t_query_med=float(np.median(tq))); res.append(s); allT.append(t_goal)
    print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ('n', 'reached', 'ratio_med', 't_goal', 't_query_med')}, flush=True)
r = np.array([x['reached'] for x in res]); rm = [x['ratio_med'] for x in res if x['ratio_med'] is not None]
S = dict(env='maze_goals', method='PRM* N=%d, K=%d целей' % (N, K), t_map=t_map, t_goal_med=float(np.median(allT)), reached_mean=float(r.mean()), ratio_med_med=float(np.median(rm)) if rm else None, total_K=t_map + float(np.sum(allT)), per_goal=res)
json.dump(S, open(os.path.join(common.DATA, 'maze_prm_goals_%d.json' % N), 'w'), indent=1, default=float); print({k: v for k, v in S.items() if k != 'per_goal'})
