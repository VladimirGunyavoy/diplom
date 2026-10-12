"""SST (Li–Littlefield–Bekris 2016) — своя реализация: pip-сборка OMPL 2.0.1 не даёт конструктор control::SST (no constructor defined). Протокол: ближайший к RRT* для динамики класс без краевого решателя.
Как в статье: выбор узла — лучший по стоимости среди активных в шаре δ_BN (selection) вокруг случайного состояния, иначе ближайший; Monte-Carlo propagation (случайное u ~ U(U), длительность 1..20 шагов по .05, без steering);
свидетели δ_s (pruning): узел остаётся активным, только если он первый или дешевле представителя свидетеля. Отклонения: батч B выборок за раунд; цель проверяется на каждом подшаге (ребро обрезается в момент входа в цель); метрика — Env.embed.
anytime: sst(...) возвращает снимки лучшего пути по бюджетам. Путь → (u_k, dt_k)."""
import time, numpy as np
from scipy.spatial import cKDTree
PSTEP = .05

def sst(e, y0, budgets, sel=.2, prune=.1, B=64, seed=0, cap=2000000, kmax=20):
    rng = np.random.default_rng(seed); n = e.n; d = e.embed_phys(np.zeros(n)).shape[0]
    Y = np.zeros((cap, n)); E = np.zeros((cap, d)); par = np.full(cap, -1); cost = np.zeros(cap); U = np.zeros((cap, e.m)); DUR = np.zeros(cap); act = np.zeros(cap, bool); Nn = 1
    Y[0] = y0; E[0] = e.embed_phys(y0); act[0] = True; Wl = np.zeros((cap, d)); Wr = np.zeros(cap, int); Wl[0] = E[0]; nW = 1
    best = np.inf; bnode = -1; snaps = []; t0 = time.perf_counter(); e.nf = 0; bi = 0; atree = None; aidx = None; dirty = True
    while bi < len(budgets):
        if dirty or atree is None: aidx = np.flatnonzero(act[:Nn]); atree = cKDTree(E[aidx]); dirty = False
        S = rng.uniform(e.lo, e.hi, (B, n)); eS = e.embed_phys(S); _, nn = atree.query(eS); ball = atree.query_ball_point(eS, sel); sl = np.empty(B, int)
        for b in range(B):
            c = ball[b]; sl[b] = aidx[c[np.argmin(cost[aidx[c]])]] if len(c) else aidx[nn[b]]
        u = rng.uniform(e.umin, e.umax, (B, e.m)); k = rng.integers(1, kmax + 1, B); Z = Y[sl].copy(); hit = np.full(B, np.inf); bad = np.zeros(B, bool); tt = np.zeros(B); live = np.ones(B, bool)
        for j in range(kmax):
            m = live & (j < k); 
            if not m.any(): break
            Z2 = e.step(Z, u, PSTEP); Z = np.where(m[:, None], Z2, Z); tt = np.where(m, tt + PSTEP, tt); bd = m & e.out_of_box(Z); bad |= bd; g = m & e.in_goal(Z) & ~bad; hit = np.where(g & np.isinf(hit), tt, hit); live &= ~(g | bd)
        dur = np.where(np.isfinite(hit), hit, tt); eZ = e.embed_phys(Z)
        for b in np.flatnonzero(~bad):
            c = cost[sl[b]] + dur[b]; dw = ((Wl[:nW] - eZ[b]) ** 2).sum(1); w = int(dw.argmin())
            if dw[w] > prune * prune: Wl[nW] = eZ[b]; Wr[nW] = Nn; nW += 1
            elif c < cost[Wr[w]] or not act[Wr[w]]: act[Wr[w]] = False; Wr[w] = Nn
            else: continue
            Y[Nn] = Z[b]; E[Nn] = eZ[b]; par[Nn] = sl[b]; cost[Nn] = c; U[Nn] = u[b]; DUR[Nn] = dur[b]; act[Nn] = True; dirty = True
            if np.isfinite(hit[b]) and c < best: best = c; bnode = Nn
            Nn += 1
        el = time.perf_counter() - t0
        while bi < len(budgets) and el >= budgets[bi]:
            if bnode >= 0:
                us, dts, i = [], [], bnode
                while par[i] >= 0: us.append(U[i].copy()); dts.append(DUR[i]); i = par[i]
                us.reverse(); dts.reverse()
            else: us = dts = None
            snaps.append(dict(budget=budgets[bi], t=el, has=bnode >= 0, T_plan=float(best) if bnode >= 0 else None, us=us, dts=dts, nodes=Nn, witnesses=nW, nf=int(e.nf))); bi += 1
        if Nn > cap - 2 * B: break
    while bi < len(budgets): snaps.append(dict(budget=budgets[bi], t=time.perf_counter() - t0, has=bnode >= 0, T_plan=None, us=None, dts=None, nodes=Nn, witnesses=nW, nf=int(e.nf))); bi += 1
    return snaps

def rrt(e, y0, budgets, B=64, seed=0, cap=2000000, kmax=20, goal_bias=.05):
    """кинодинамический RRT (LaValle–Kuffner 2001), как control::RRT в OMPL: ближайший узел к выборке (goal bias .05 — выборка из цели), случайное u, длительность 1..20·.05, без steering;
    отличие от pip-OMPL: цель проверяется на каждом подшаге (ребро обрезается при входе). Останавливается на первом решении; снимки по бюджетам = то же решение (если найдено к бюджету)."""
    rng = np.random.default_rng(seed); n = e.n; d = e.embed_phys(np.zeros(n)).shape[0]
    Y = np.zeros((cap, n)); E = np.zeros((cap, d)); par = np.full(cap, -1); cost = np.zeros(cap); U = np.zeros((cap, e.m)); DUR = np.zeros(cap); Nn = 1; Y[0] = y0; E[0] = e.embed_phys(y0)
    t0 = time.perf_counter(); e.nf = 0; tree = None; tN = 0; found = None; tfound = None
    while time.perf_counter() - t0 < budgets[-1] and found is None and Nn < cap - 2 * B:
        S = rng.uniform(e.lo, e.hi, (B, n)); gb = rng.random(B) < goal_bias; S[gb] = 0.; eS = e.embed_phys(S)
        if tree is None or Nn > 1.3 * tN + 100: tree = cKDTree(E[:Nn]); tN = Nn
        dd, nn = tree.query(eS)
        if Nn > tN: Bf = E[tN:Nn]; d2 = ((eS[:, None, :] - Bf[None]) ** 2).sum(-1); j = d2.argmin(1); m = d2[np.arange(B), j] < dd ** 2; nn = np.where(m, tN + j, nn)
        u = rng.uniform(e.umin, e.umax, (B, e.m)); k = rng.integers(1, kmax + 1, B); Z = Y[nn].copy(); hit = np.full(B, np.inf); bad = np.zeros(B, bool); tt = np.zeros(B); live = np.ones(B, bool)
        for j in range(kmax):
            m = live & (j < k)
            if not m.any(): break
            Z2 = e.step(Z, u, PSTEP); Z = np.where(m[:, None], Z2, Z); tt = np.where(m, tt + PSTEP, tt); bd = m & e.out_of_box(Z); bad |= bd; g = m & e.in_goal(Z) & ~bad; hit = np.where(g & np.isinf(hit), tt, hit); live &= ~(g | bd)
        dur = np.where(np.isfinite(hit), hit, tt); eZ = e.embed_phys(Z); ok = np.flatnonzero(~bad); a = Nn; c = len(ok)
        Y[a:a + c] = Z[ok]; E[a:a + c] = eZ[ok]; par[a:a + c] = nn[ok]; cost[a:a + c] = cost[nn[ok]] + dur[ok]; U[a:a + c] = u[ok]; DUR[a:a + c] = dur[ok]
        gh = np.flatnonzero(np.isfinite(hit[ok]))
        if len(gh): found = a + gh[0]; tfound = time.perf_counter() - t0
        Nn += c
    el = time.perf_counter() - t0; snaps = []
    if found is not None:
        us, dts, i = [], [], found
        while par[i] >= 0: us.append(U[i].copy()); dts.append(DUR[i]); i = par[i]
        us.reverse(); dts.reverse()
    for B_ in budgets:
        ok = found is not None and tfound <= B_
        snaps.append(dict(budget=B_, t=(tfound if ok else min(B_, el)), has=ok, T_plan=float(cost[found]) if ok else None, us=us if ok else None, dts=dts if ok else None, nodes=Nn, witnesses=0, nf=int(e.nf)))
    return snaps
