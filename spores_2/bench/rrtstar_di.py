"""kinodynamic RRT* для ДИ с ТОЧНЫМ мин.-временным краевым (Webb–van den Berg 2013). Краевая замкнута: из (x,a) в (x+D,b) bang-bang, T₊ = 2√((a²+b²)/2 + D) − a − b (сначала +1, затем −1; допустимо при √ ≥ max(a,b)),
T₋ = 2√((a²+b²)/2 − D) + a + b (сначала −1; √ ≥ max(−a,−b)). Окрестность — по времени краевой T(j→new) ≤ r_T = γ·(log n / n)^{1/4}, γ=2 (подбор), перепривязка с обновлением стоимостей потомков.
Цель: выборки с вероятностью .05 из круга цели; узел внутри круга — решение. Снимки лучшего пути по бюджетам (anytime)."""
import time, numpy as np

def bvp(x0, v0, x1, v1):
    """векторно: T, s (первое управление ±1), t1, t2; inf если нет"""
    D = x1 - x0; R = (v0 * v0 + v1 * v1) / 2
    rp = R + D; rm = R - D; sp = np.sqrt(np.maximum(rp, 0)); sm = np.sqrt(np.maximum(rm, 0))
    okp = (rp >= 0) & (sp >= np.maximum(v0, v1)); okm = (rm >= 0) & (sm >= np.maximum(-v0, -v1))
    Tp = np.where(okp, 2 * sp - v0 - v1, np.inf); Tm = np.where(okm, 2 * sm + v0 + v1, np.inf)
    usep = Tp <= Tm; T = np.where(usep, Tp, Tm)
    t1 = np.where(usep, sp - v0, sm + v0); t2 = np.where(usep, sp - v1, sm + v1)
    return T, np.where(usep, 1., -1.), t1, t2

def arcs_ok(x0, v0, s, t1, t2, hi):
    """проверка коробки вдоль 2 дуг (по 8 точкам каждой)"""
    for tt in np.linspace(0, 1, 9)[1:]:
        a = t1 * tt; x = x0 + v0 * a + s * a * a / 2; v = v0 + s * a
        if (np.abs(x) > hi[0]).any() or (np.abs(v) > hi[1]).any(): return False
    xm = x0 + v0 * t1 + s * t1 * t1 / 2; vm = v0 + s * t1
    for tt in np.linspace(0, 1, 9)[1:]:
        b = t2 * tt; x = xm + vm * b - s * b * b / 2; v = vm - s * b
        if (np.abs(x) > hi[0]).any() or (np.abs(v) > hi[1]).any(): return False
    return True

def rrtstar(e, y0, budgets, seed=0, gamma=2., goal_bias=.05, cap=60000):
    rng = np.random.default_rng(seed); X = np.zeros((cap, 2)); par = np.full(cap, -1); cost = np.zeros(cap); X[0] = y0; N = 1; kids = {0: []}; t0 = time.perf_counter(); e.nf = 0
    best = np.inf; bnode = -1; snaps = []; bi = 0; goal_nodes = []
    def chain(i):
        us, dts = [], []
        path = []
        while par[i] >= 0: path.append(i); i = par[i]
        path.reverse()
        for k in path:
            p = par[k]; T, s, t1, t2 = bvp(X[p, 0], X[p, 1], X[k, 0], X[k, 1]); us += [[s], [-s]]; dts += [float(t1), float(t2)]
        return us, dts
    def upd(i, delta):
        st = [i]
        while st:
            k = st.pop(); cost[k] += delta; st.extend(kids[k])
    while bi < len(budgets):
        for _ in range(20):
            if rng.random() < goal_bias:
                r = e.ball * np.sqrt(rng.random()) * .98; th = rng.random() * 2 * np.pi; xn = np.array([r * np.cos(th), r * np.sin(th)])
            else: xn = rng.uniform(e.lo, e.hi)
            rT = gamma * (np.log(N + 1) / (N + 1)) ** .25 * 4; e.nf += 1
            Tin, s, t1, t2 = bvp(X[:N, 0], X[:N, 1], xn[0], xn[1])
            near = np.flatnonzero(Tin <= rT)
            if len(near) == 0:
                j = int(np.argmin(Tin))
                if not np.isfinite(Tin[j]): continue
                near = np.array([j])
            tot = cost[near] + Tin[near]; order = np.argsort(tot); pb = -1
            for k in order:
                j = near[k]
                if np.isfinite(tot[k]) and arcs_ok(X[j, 0], X[j, 1], s[j], t1[j], t2[j], e.hi): pb = j; cb = tot[k]; break
            if pb < 0: continue
            if N >= cap - 1: break
            n_ = N; X[n_] = xn; par[n_] = pb; cost[n_] = cb; kids[n_] = []; kids[pb].append(n_); N += 1
            To, so, a1, a2 = bvp(xn[0], xn[1], X[near, 0], X[near, 1])                              # перепривязка: new → near
            for k, j in enumerate(near):
                if j == pb or j == 0: continue
                if cost[n_] + To[k] < cost[j] - 1e-9 and arcs_ok(xn[0], xn[1], so[k], a1[k], a2[k], e.hi):
                    kids[par[j]].remove(j); upd(j, cost[n_] + To[k] - cost[j]); par[j] = n_; kids[n_].append(j)
            if e.in_goal(xn[None])[0] and cost[n_] < best: best = cost[n_]; bnode = n_
            elif bnode >= 0 and e.in_goal(X[bnode][None])[0]: best = cost[bnode]
        el = time.perf_counter() - t0
        while bi < len(budgets) and el >= budgets[bi]:
            if bnode >= 0: us, dts = chain(bnode)
            else: us = dts = None
            snaps.append(dict(budget=budgets[bi], t=el, has=bnode >= 0, T_plan=float(cost[bnode]) if bnode >= 0 else None, us=us, dts=dts, nodes=N, witnesses=0, nf=int(e.nf))); bi += 1
    return snaps
