"""Граф V по клеткам (PLAN 0з п.3): рёбра клетка C → клетка D (того же слоя — выход, другого — переключение) в момент t_j вдоль центральной траектории C
(точка X(t_j) принадлежит ядру/гало D по модели, без rk4); цена ребра — время t_j. Цель: клетки, содержащие цель (цена t_цель). V — Дейкстра от цели назад."""
import heapq
import numpy as np

NS = 9


def _entry_pts(D, ns=3, nt=9, lat=1.0):
    """Точки ядра D (сетка s×t) и их время t внутри D по модели."""
    s = np.linspace(-1, 1, ns) * lat * D.r; t = np.linspace(0, D.tau, nt) * 0.95
    X = np.array([np.interp(t, D.ts, D.X[:, i]) for i in range(2)]).T; V = np.array([np.interp(t, D.ts, D.V[:, i]) for i in range(2)]).T
    return (X[None] + s[:, None, None] * V[None]).reshape(-1, 2), np.tile(t, ns)


def build_graph(S, layers, goal, halo=1.1, ns=3, nt=9, lat=1.0):
    """Узел = (k, i). Ребро C → D: точка y из ядра D лежит в C (по модели, момент t_C внутри C) — из C в момент t_C можно перейти в D, оказавшись
    в D в момент g = t_D(y). edges[C] = [(t_C, g, D)]; цена перехода из C: t_C + V(D) − g, допустимо при g ≤ времени действия D. Одна точка → все содержащие её клетки (оба слоя, в т.ч. свой)."""
    nodes = [(k, i) for k in range(len(layers)) for i in range(len(layers[k]))]; edges = {n: [] for n in nodes}; goal_t = {}
    cen = {n: layers[n[0]][n[1]] for n in nodes}
    rad = {n: float(np.max(np.linalg.norm(S.wrap(C.X - C.c), axis=1)) + C.r * float(np.max(np.linalg.norm(C.V, axis=1))) + 1e-3) for n, C in cen.items()}
    ctr = np.array([cen[n].c for n in nodes]); rd = np.array([rad[n] for n in nodes])
    for jd, nd in enumerate(nodes):
        D = cen[nd]; Y, gD = _entry_pts(D, ns, nt, lat)
        near = np.nonzero(np.linalg.norm(S.wrap(ctr - D.c), axis=1) <= rd + rad[nd])[0]
        for jc in near:
            nc = nodes[jc]
            if nc == nd: continue
            sC, tC, ins = cen[nc].locate(Y, halo)
            if not ins.any(): continue
            q = np.nonzero(ins & (np.abs(sC) <= lat * cen[nc].r))[0]
            for qq in q: edges[nc].append((max(float(tC[qq]), 0.0), float(gD[qq]), nd))
    g = np.array(goal, float)
    for n, C in cen.items():
        s_, t_, ins = C.locate(g, halo)
        if ins[0]: goal_t[n] = float(t_[0])
    return edges, goal_t


def solve_V(layers, edges, goal_t):
    """V(узел) = время от центра клетки (t=0) до цели; Дейкстра назад. Вес ребра (t_C, g, D): t_C + max(V(D) − g, 0) — зависит от V(D), но монотонен."""
    rev = {}
    for n, es in edges.items():
        for tc, g, m in es: rev.setdefault(m, []).append((tc, g, n))
    V = {}; Ts = {}; h = [(max(t, 0.0), n, max(t, 0.0)) for n, t in goal_t.items()]; heapq.heapify(h)
    while h:
        v, n, ta = heapq.heappop(h)
        if n in V: continue
        V[n] = v; Ts[n] = ta                                  # ta — время действия клетки (цель / переключение): войти в неё можно лишь до ta
        for tc, g, p in rev.get(n, []):
            if p not in V and g <= ta + 1e-9: heapq.heappush(h, (tc + v - g, p, tc))
    solve_V.Ts = Ts
    return V


def query(S, layers, edges, goal_t, V, y, halo=1.0):
    """Оценка V в точке y: по клеткам, содержащим y (t_loc), минимум по рёбрам с t_j ≥ t_loc: (t_j − t_loc) + V(цель ребра)."""
    best = np.inf
    for k, L in enumerate(layers):
        for i, C in enumerate(L):
            s, t, ins = C.locate(y, halo)
            if not ins[0]: continue
            tl = max(float(t[0]), 0.0); n = (k, i)
            if n in goal_t and goal_t[n] >= tl: best = min(best, goal_t[n] - tl)
            for tc, g, m in edges[n]:
                if tc >= tl and m in V and g <= solve_V.Ts[m] + 1e-9: best = min(best, tc - tl + V[m] - g)
    return best
