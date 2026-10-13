"""CA-MAZE 4D: PRM* на (x, y) (карта один раз, как prm_maze.py) + контроллер преследования пути для дд с инерцией (v̇ = a − v): запрос = путь по карте + замкнутое моделирование (RK4 h=.01, тот же env.f)
→ последовательность (u, dt) → единый проверяльщик. Время запроса включает моделирование. python prm_maze4.py [N=3000] [MARG=.06] [LD=.3] → data/maze4_prms_N.json"""
import os, sys, time, json, heapq, numpy as np
from scipy.spatial import cKDTree
import common, maze4
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; MARG = float(sys.argv[2]) if len(sys.argv) > 2 else .06; LD = float(sys.argv[3]) if len(sys.argv) > 3 else .3; H = .01
e = common.make('maze4'); M = e.M; rng = np.random.default_rng(0)
def los(a, b):
    t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / .01)))[:, None]; return (M.gfree(a + t * (b - a)) >= MARG - 1e-9).all()
t0 = time.perf_counter(); P = [np.zeros(2)]
while len(P) < N:
    p = rng.uniform(0, M.n, 2) - M.G0
    if M.gfree(p[None])[0] >= MARG: P.append(p)
P = np.array(P); tree = cKDTree(P); adj = [[] for _ in range(N)]; r = 2.0 * np.sqrt(np.log(N) / N) * M.n * 1.6
for i in range(N):
    for j in tree.query_ball_point(P[i], r):
        if j > i and los(P[i], P[j]): w = float(np.linalg.norm(P[i] - P[j])); adj[i].append((j, w)); adj[j].append((i, w))
d = np.full(N, np.inf); d[0] = 0.; par = -np.ones(N, int); pq = [(0., 0)]
while pq:
    c, u = heapq.heappop(pq)
    if c > d[u]: continue
    for v, w in adj[u]:
        if c + w < d[v]: d[v] = c + w; par[v] = u; heapq.heappush(pq, (d[v], v))
t_build = time.perf_counter() - t0
def query(y0):
    a = y0[:2]; cand = [j for j in tree.query(a, 30)[1] if np.isfinite(d[j]) and los(a, P[j])]
    if not cand: return None
    j = min(cand, key=lambda j: np.linalg.norm(a - P[j]) + d[j]); path = [a]
    while j != -1: path.append(P[j]); j = par[j]
    Q = [path[0]]; i = 0                                  # натяжение пути (прямая видимость)
    while i < len(path) - 1:
        k = len(path) - 1
        while k > i + 1 and not los(path[i], path[k]): k -= 1
        Q.append(path[k]); i = k
    return np.array(Q)
def follow(y0, Q, tmax=60.):
    """преследование: цель — точка на пути в LD впереди проекции; ω ∝ ошибке курса, a ведёт v к v_d = cos(ошибки)⁺; |a| + |ω| ≤ 1"""
    seg = np.diff(Q, axis=0); L = np.linalg.norm(seg, axis=1); S = np.r_[0, np.cumsum(L)]; y = np.asarray(y0, float)[None].copy(); us = []; dts = []; s_prev = 0.
    for _ in range(int(tmax / H)):
        p = y[0, :2]; k = min(max(np.searchsorted(S, s_prev, 'right') - 1, 0), len(seg) - 1)
        best = None
        for kk in range(k, min(k + 6, len(seg))):                      # проекция на ближайшие звенья
            t = np.clip(np.dot(p - Q[kk], seg[kk]) / max(L[kk] ** 2, 1e-12), 0, 1); dist = np.linalg.norm(p - (Q[kk] + t * seg[kk]))
            if best is None or dist < best[0]: best = (dist, S[kk] + t * L[kk])
        s_prev = max(s_prev, best[1]); st = min(s_prev + LD, S[-1]); kk = min(max(np.searchsorted(S, st, 'right') - 1, 0), len(seg) - 1)
        tgt = Q[kk] + (st - S[kk]) / max(L[kk], 1e-12) * seg[kk]; ang = np.arctan2(tgt[1] - p[1], tgt[0] - p[0]); al = float(np.angle(np.exp(1j * (ang - y[0, 2]))))
        w = float(np.clip(3. * al, -1, 1)); vd = max(np.cos(al), 0.) * (1 - abs(w)) ; a = float(np.clip(y[0, 3] + 3. * (vd - y[0, 3]), -(1 - abs(w)), 1 - abs(w)))
        u = np.array([[a, w]]); y = e.step(y, u, H); us.append([a, w]); dts.append(H)
        if e.in_goal(y)[0] or e.out_of_box(y)[0]: break
    return us, dts
Ts = []; tq = []; res = []
for i, y0 in enumerate(e.X0):
    t1 = time.perf_counter(); Q = query(y0)
    if Q is None: dtq = time.perf_counter() - t1; Tv, why = None, 'no_path'
    else: us, dts = follow(y0, Q); dtq = time.perf_counter() - t1; Tv, why = common.verify(e, y0, us, dts)
    Ts.append(Tv); tq.append(dtq); res.append(dict(i=i, T_verify=Tv, why=why, t_query=dtq))
s = common.summarize(e, Ts); s.update(env='maze4', method='PRM* N=%d MARG=%.2f LD=%.2f + преследование' % (N, MARG, LD), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)), hw='CPU 1 proc, numpy',
                                      weight=dict(bytes=int(P.nbytes + sum(len(a) for a in adj) * 16)), per_start=res, Tref=e.Tref.tolist())
json.dump(s, open(os.path.join(common.DATA, 'maze4_prms_%d.json' % N), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}); print([r['why'] for r in res if r['why'] != 'ok'][:10])
