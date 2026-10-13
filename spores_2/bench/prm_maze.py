"""CA-MAZE: PRM / PRM* для дд в лабиринте. Карта один раз (узлы в свободном (x,y), рёбра по прямой видимости gfree ≥ запас), запрос = соединение старта с k ближайшими видимыми узлами + Dijkstra; путь исполняется rotate–straight–rotate
(точный шаг дд: T = Σ длин + Σ |поворотов|, начальный курс y0[2], конечный 0). python prm_maze.py [N=3000] [star=0|1]  → data/maze_prm[star]_N.json"""
import os, sys, time, json, heapq, numpy as np
from scipy.spatial import cKDTree
import common, maze
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; STAR = int(sys.argv[2]) if len(sys.argv) > 2 else 0; K = 15; MARG = .02
e = common.make('maze'); M = e.M; rng = np.random.default_rng(0)
def los(a, b):
    t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / .01)))[:, None]; return (M.gfree(a + t * (b - a)) >= MARG - 1e-9).all()
t0 = time.perf_counter()
P = [np.zeros(2)]
while len(P) < N:
    p = rng.uniform(0, M.n, 2) - M.G0
    if M.gfree(p[None])[0] >= MARG: P.append(p)
P = np.array(P); tree = cKDTree(P); adj = [[] for _ in range(N)]
r = 2.0 * np.sqrt(np.log(N) / N) * M.n * (1.6 if STAR else 1.) if STAR else None
for i in range(N):
    nb = tree.query_ball_point(P[i], r) if STAR else tree.query(P[i], K + 1)[1]
    for j in nb:
        if j > i or (not STAR and j != i):
            if j != i and i not in [a for a, _ in adj[j]] and los(P[i], P[j]): w = float(np.linalg.norm(P[i] - P[j])); adj[i].append((j, w)); adj[j].append((i, w))
t_build = time.perf_counter() - t0
# расстояния от цели (узел 0) — один Dijkstra, дальше запрос = выбор лучшего соседа
t0 = time.perf_counter(); d = np.full(N, np.inf); d[0] = 0.; par = -np.ones(N, int); pq = [(0., 0)]
while pq:
    c, u = heapq.heappop(pq)
    if c > d[u]: continue
    for v, w in adj[u]:
        if c + w < d[v]: d[v] = c + w; par[v] = u; heapq.heappush(pq, (d[v], v))
t_build += time.perf_counter() - t0
def query(y0):
    a = y0[:2]; cand = [j for j in tree.query(a, 30)[1] if np.isfinite(d[j]) and los(a, P[j])]
    if not cand: return None
    j = min(cand, key=lambda j: np.linalg.norm(a - P[j]) + d[j]); path = [a]
    while j != -1: path.append(P[j]); j = par[j]
    return np.array(path)
def execute(y0, Q):
    seg = np.diff(Q, axis=0); L = np.linalg.norm(seg, axis=1); hd = np.arctan2(seg[:, 1], seg[:, 0]); h = np.r_[y0[2], hd, 0.]; us = []; dts = []
    for k in range(len(seg)):
        da = float(np.angle(np.exp(1j * (h[k + 1] - h[k])))); 
        if abs(da) > 1e-9: us.append([0., np.sign(da)]); dts.append(abs(da))
        us.append([1., 0.]); dts.append(L[k])
    da = float(np.angle(np.exp(1j * (h[-1] - h[-2]))))
    if abs(da) > 1e-9: us.append([0., np.sign(da)]); dts.append(abs(da))
    return us, dts
Ts = []; tq = []; res = []
for i, y0 in enumerate(e.X0):
    t1 = time.perf_counter(); Q = query(y0); dtq = time.perf_counter() - t1
    if Q is None: Tv, why = None, 'no_path'
    else:
        us, dts = execute(y0, Q); Tv, why = common.verify(e, y0, us, dts)
    Ts.append(Tv); tq.append(dtq); res.append(dict(i=i, T_verify=Tv, why=why, t_query=dtq))
s = common.summarize(e, Ts); s.update(env='maze', method='PRM%s N=%d' % ('*' if STAR else '', N), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)), hw='CPU 1 proc, numpy',
                                      weight=dict(bytes=int(P.nbytes + sum(len(a) for a in adj) * 16)), per_start=res, Tref=e.Tref.tolist())
json.dump(s, open(os.path.join(common.DATA, 'maze_prm%s_%d.json' % ('s' if STAR else '', N)), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
