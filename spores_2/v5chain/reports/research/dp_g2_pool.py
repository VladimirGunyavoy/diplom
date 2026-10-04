"""research-11: «пул траекторий» — путь агента FORCEPLAN (реальная траектория, цена точная = остаток времени) кладётся в атлас спорами
(точки дуг через ≤ DT с, нормаль переносится назад по дуге от конца), пары — обычным образом, затем агент идёт снова: мини-дерево цепляется
за споры прошлого пути и срезает. Концепция пользователя: каждый запрос — своё прямое дерево в общий пул. Без эталона внутри.
Запуск: G=2 WIN=1.0 FORCEPLAN=1 WCHK=1 ITERS=6 python3 dp_g2_pool.py atlas.npz"""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, BIG, WIN
from butterfly_dp_atlas import R, MN
from scipy.spatial import cKDTree
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
from butterfly_dp_grow import emb, normals
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; KS = 81; t0 = time.time(); DT = float(os.environ.get('DT', .3)); OCP = float(os.environ.get('OCP', 7.636)); rng = np.random.default_rng(0)
for it in range(int(os.environ.get('ITERS', 6)) + 1):
    Q.TRAJ.clear(); A.E = Q.node_edges(A); A.nplan = 0; T, arcs, wm = Q.rollout_edges(A, KS, 0.)
    print(json.dumps(dict(iter=it, spores=int(A.K), V=round(float(A.V[KS, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / OCP), 4), arcs=arcs, wmax=round(float(wm), 2), sec=round(time.time() - t0))), flush=True)
    if not np.isfinite(T) or it == int(os.environ.get('ITERS', 6)): break
    pts, seg = [], []                                                                         # точки пути и (u1, u2, dt) до следующей
    for y, u1, u2, tt in Q.TRAJ:
        m = max(1, int(np.ceil(tt / DT))); z = y.copy()
        for _ in range(m): pts.append(z.copy()); seg.append((u1, u2, tt / m)); z = flow(z[:, None], np.array([u1]), np.array([u2]), tt / m, n=12)[:, 0]
    pts.append(z.copy()); P = np.array(pts); nn = np.zeros_like(P); nn[-1] = normals(P[-1:], rng)[0]
    for i in range(len(P) - 2, -1, -1):                                                      # перенос нормали назад: узлы споры i ложатся на отрезок споры i + 1
        u1, u2, dt = seg[i]; d = (flow((P[i + 1] + 1e-5 * nn[i + 1])[:, None], np.array([u1]), np.array([u2]), -dt, n=12)[:, 0] - P[i]) / 1e-5; nn[i] = d / np.linalg.norm(d)
    P[:, :2] = wrap(P[:, :2]); dd, jj = cKDTree(A.X).query(emb(P)); keep = dd > .01; K0 = A.K; idx = np.where(keep, K0 + np.cumsum(keep) - 1, jj); dts = np.array([x[2] for x in seg]); P, nn = P[keep], nn[keep]
    A.C = np.r_[A.C, P]; A.n = np.r_[A.n, nn]; A.K = len(A.C); A.V = np.r_[A.V, np.full((A.K - K0, MN), BIG)]; E.refresh(A); A.V[A.ingoal] = 0.
    Yn = A.P[K0:].reshape(-1, 4); idn = np.arange(K0 * MN, A.K * MN); own = np.repeat(np.arange(K0, A.K), MN); e_new = E.edges_add(A, idn, A.pairs(Yn, own))
    near = np.unique(np.concatenate([np.asarray(b, int) for b in cKDTree(A.X[:K0]).query_ball_point(A.X[K0:], 2 * WIN)] or [np.zeros(0, int)]))
    Yo = A.P[near].reshape(-1, 4); ido = (near[:, None] * MN + np.arange(MN)).reshape(-1); out = A.pairs(Yo); kp = out[1].astype(int) >= K0; e_old = E.edges_add(A, ido, [x[kp] for x in out])
    m = len(dts); e_ch = [idx[:-1] * MN + MN // 2, idx[1:], np.full(m, MN // 2), np.zeros(m), dts]                 # точные рёбра самого пути: центр i → центр i + 1 (известные u, dt)
    e = [np.r_[a, b, c, d] for a, b, c, d in zip(A.e, e_new, e_old, e_ch)]; o = np.argsort(e[0], kind='stable'); A.e = [x[o] for x in e]; A.solve()
    print(json.dumps(dict(pooled=int(A.K - K0), new_centers_fin=int((A.V[K0:, MN // 2] < BIG / 2).sum()), V=round(float(A.V[KS, MN // 2]), 3))), flush=True)
np.savez(sys.argv[1].replace('.npz', '_pool.npz'), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
