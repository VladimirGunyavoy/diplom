"""RRT* для дифдрайва с ромбовым U (|v|+|ω| ≤ 1) с точным локальным планировщиком Balkcom–Mason (Karaman–Frazzoli 2011: ребро и соединение с целью — кривая планировщика).
Локальный планировщик: TGT (поворот на месте — прямая — поворот, вперёд/задом; замкнутая форма, оптимум BM для большинства поз). Для боковых сдвигов оптимум BM — 5 отрезков (TGTGT, до −37% к TGT); здесь они получаются через промежуточные узлы дерева (отклонение, помечено в таблице).
Дерево растёт от СТАРТА к цели (одиночный запрос). Интерфейс как rrtstar_di.rrtstar: snaps по бюджетам [(budget, t, has, T_plan, us, dts, nodes, nf, hist)]."""
import time, numpy as np
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
def tgt_cost(P, q):
    """стоимость и параметры TGT от поз P (n,3) до позы q (3,) или от позы P (3,) до набора q (n,3); возвращает (cost, back, h, d)"""
    P = np.atleast_2d(P); q = np.atleast_2d(q); dx = q[:, 0] - P[:, 0]; dy = q[:, 1] - P[:, 1]; d = np.hypot(dx, dy); phi = np.arctan2(dy, dx)
    h0 = phi; h1 = wrap(phi + np.pi)
    c0 = np.abs(wrap(h0 - P[:, 2])) + d + np.abs(wrap(q[:, 2] - h0)); c1 = np.abs(wrap(h1 - P[:, 2])) + d + np.abs(wrap(q[:, 2] - h1))
    back = c1 < c0; c = np.where(back, c1, c0); h = np.where(back, h1, h0)
    tiny = d < 1e-12; c = np.where(tiny, np.abs(wrap(q[:, 2] - P[:, 2])), c)
    return c, back, h, d
def seg(p, q):
    """управления (u=(v,ω), dt) для перехода p→q по TGT"""
    c, back, h, d = tgt_cost(p, q); back = bool(back[0]); h = float(h[0]); d = float(d[0]); us = []; dts = []
    if d < 1e-12: a = float(wrap(q[2] - p[2])); return [[0., np.sign(a) or 1.]], [abs(a)]
    a1 = float(wrap(h - p[2])); a2 = float(wrap(q[2] - h))
    if abs(a1) > 1e-12: us.append([0., np.sign(a1)]); dts.append(abs(a1))
    us.append([-1. if back else 1., 0.]); dts.append(d)
    if abs(a2) > 1e-12: us.append([0., np.sign(a2)]); dts.append(abs(a2))
    return us, dts
def rrtstar(e, y0, budgets, seed=0, goal_bias=.05, cap=40000, gam=2.):
    rng = np.random.default_rng(seed); X = np.zeros((cap, 3)); par = np.full(cap, -1); cost = np.zeros(cap); X[0] = y0; N = 1; kids = {0: []}; t0 = time.perf_counter(); e.nf = 0
    goal = np.zeros(3); snaps = []; bi = 0; hist = []; lastbest = np.inf
    def best():
        c, *_ = tgt_cost(X[:N], goal); tot = cost[:N] + c; j = int(np.argmin(tot)); return j, float(tot[j])
    def chain(j):
        path = []; i = j
        while par[i] >= 0: path.append(i); i = par[i]
        path.reverse(); us, dts = [], []
        for k in path: u, t = seg(X[par[k]], X[k]); us += u; dts += t
        u, t = seg(X[j], goal); return us + u, dts + t
    def upd(i, delta):
        st = [i]
        while st:
            k = st.pop(); cost[k] += delta; st.extend(kids[k])
    while bi < len(budgets):
        for _ in range(20):
            if N >= cap - 1: break
            xn = goal.copy() if rng.random() < goal_bias else np.array([rng.uniform(-2.2, 2.2), rng.uniform(-2.2, 2.2), rng.uniform(-np.pi, np.pi)]); e.nf += 1
            Tin, *_ = tgt_cost(X[:N], xn); k = min(N, int(np.ceil(2 * np.e * np.log(N + 1))) + 1); near = np.argpartition(Tin, k - 1)[:k] if k < N else np.arange(N)
            tot = cost[near] + Tin[near]; pb = int(near[np.argmin(tot)]); cb = float(tot.min()); n_ = N; X[n_] = xn; par[n_] = pb; cost[n_] = cb; kids[n_] = []; kids[pb].append(n_); N += 1
            To, *_ = tgt_cost(xn, X[near])                                                         # перепривязка: new → near
            for kk, j in enumerate(near):
                if j == pb or j == 0: continue
                if cost[n_] + To[kk] < cost[j] - 1e-9:
                    kids[par[j]].remove(j); upd(j, cost[n_] + To[kk] - cost[j]); par[j] = n_; kids[n_].append(j)
        el = time.perf_counter() - t0
        j, bc = best()
        if bc < lastbest - 1e-12: hist.append((el, bc)); lastbest = bc
        while bi < len(budgets) and el >= budgets[bi]:
            us, dts = chain(j); snaps.append(dict(budget=budgets[bi], t=el, has=True, T_plan=bc, us=us, dts=dts, nodes=N, witnesses=0, nf=int(e.nf), hist=list(hist))); bi += 1
    return snaps
