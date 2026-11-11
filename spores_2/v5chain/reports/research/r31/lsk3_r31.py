import os, sys, pickle, json, time
os.environ.update(SYS='car', M='3', SIDE='0', GM='.25', GLIM='0', FRAC='1', PESS='1', GS='300', OMP_NUM_THREADS='1')
sys.path.insert(0, '.'); sys.path.insert(0, '../v7/src/cells7')
import numpy as np, envs_dyn as D, growCar as G, __main__, inspect, numba as nb
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(sys.argv[1]), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
P = A.P; V0 = A.V.copy(); BIG = G.BIG; DTN = G.DTN; US = G.US; kap = np.sqrt(-2*np.log(.03)); T_PR = float(os.environ.get('TPR', 2.))
res = json.load(open('/home/aida/spore_v5/reports/bench/data/env_A_fixed_lk0_H1.5.json'))['res']
# агент: точка на траектории (середина эпизода 3)
cand = [np.array(r['Y'])[k] for r in res[:12] for k in (2, 8, 14) if k < len(r['Y'])]; vv = A.vstar(np.array(cand)); ok = [(c, v) for c, v in zip(cand, vv) if v < BIG / 2 and v > 3.]
ok.sort(key=lambda t: np.linalg.norm(t[0][:2] - D.C0[0])); y, va = ok[0]; print('агент', y, 'V0', va, 'кандидатов', len(ok))
dist = np.linalg.norm(P[:, None, :2] - D.C0[None], axis=2).min(1); Dr = D.HR + kap*.25 + .3
reg = np.flatnonzero((dist <= Dr) & (V0 <= va) & (va - V0 <= T_PR)); print('область', len(reg), 'узлов')
t0 = time.time(); Is, Xs, Ws, Vg = [], [], [], np.full(len(reg), np.inf)
for u in US:
    Vg = np.minimum(Vg, A.tgoal(P[reg], u)); a, b, c = A.stencils(G.step(P[reg], u)); Is.append(a); Xs.append(b.astype(np.int64)); Ws.append(c)
I = np.concatenate(Is); X = np.concatenate(Xs); W = np.concatenate(Ws); o = np.argsort(I, kind='stable'); I, X, W = I[o], X[o], W[o]
st = np.r_[0, np.flatnonzero(I[1:] != I[:-1]) + 1, len(I)]; nd = I[st[:-1]]
print('рёбра области: %d, построение %.1f с (один раз)' % (len(I), time.time() - t0))
order = np.argsort(V0[reg][nd]); WTHR = G.WTHR
@nb.njit(cache=True)
def sweep(V, nd_g, st, X, W, Vg_n, order, BIG, DTN, WTHR, forb, pe):
    ch = 0.
    for oi in range(len(order)):
        k = order[oi]; i = nd_g[k]
        if forb[i]: continue
        best = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            sm = 0.; s = 0.; mx = -1e30
            for j in range(W.shape[1]):
                w = W[e, j]; v = V[X[e, j]]
                if v < BIG / 2:
                    s += w * v; sm += w
                    if v > mx: mx = v
            if sm < WTHR: continue
            val = DTN + s + (1 - sm) * (mx + pe)
            if val < best: best = val
        if best != V[i]:
            ch = max(ch, abs(best - V[i]) if best < BIG / 2 and V[i] < BIG / 2 else 1.)
            V[i] = best
    return ch
Vg_n = Vg[nd]; Vg_full = np.full(len(P), BIG)
# запрет: диск стоит в C0 (w=0), раздутый на κσ(τ), τ = V0(агент) − V0(узел), σ ≤ .25
tau = np.maximum(va - V0[reg], 0.); sg = np.minimum(np.sqrt(.01 * tau ** 3 / 3), .25)
fm = (np.linalg.norm(P[reg][:, None, :2] - D.C0[None], axis=2) - D.HR - .1 - kap * sg[:, None]).min(1) < 0
forb = np.zeros(len(P), bool); forb[reg[fm]] = True; print('запрещено узлов', int(fm.sum()), 'из', len(reg))
V = V0.copy(); V[reg[fm]] = BIG
Vg_n = np.where(np.isfinite(Vg_n), Vg_n, BIG)
# ===== research-31: однопроходной label-setting (Дейкстра с инкрементными суммами рёбер) против GS-до-сходимости =====
import heapq
from numba import types
from numba.typed import List
forb_g = forb.copy()
goal_l = A.goal[reg]
def gs_ref():
    V = V0.copy(); V[reg] = BIG; V[reg[goal_l]] = 0.; fx = forb_g.copy(); fx[reg[goal_l]] = True
    t0 = time.time(); n = 0; chs = []
    while n < 120:
        n += 1; order = np.argsort(V[reg[nd]], kind='stable'); ch = sweep(V, reg[nd], st, X, W, Vg_n, order, BIG, DTN, WTHR, fx, 1.)
        chs.append(ch)
        if ch < 1e-6: break
    return V, time.time() - t0, n, chs
V_gs, t_gs, n_gs, chs_gs = gs_ref()
A.V = V_gs.copy(); vg = float(A.vstar(y[None])[0]); A.V = V0.copy()
print('GS (сорт. по тек. V) до ch<1e-6: %.2f с, проходов %d, ch конец %.2e' % (t_gs, n_gs, chs_gs[-1]))
# --- подготовка LS (зависит только от области; при статичной области — один раз)
t0 = time.time()
nreg = len(reg); loc = np.full(len(P), -1, np.int64); loc[reg] = np.arange(nreg)
E = len(X); K = W.shape[1]
own = np.repeat(nd, np.diff(st)).astype(np.int64)            # локальный узел-владелец ребра
Xl = loc[X]; inreg = (Xl >= 0) & (W > 1e-12)
ee, jj = np.nonzero(inreg); key = Xl[ee, jj]; o = np.argsort(key, kind='stable'); ee = ee[o].astype(np.int64); ww = W[ee, jj[o]].astype(np.float64)
rst = np.searchsorted(key[o], np.arange(nreg + 1))
print('LS подготовка (обратный индекс %d вхождений): %.1f с' % (len(ee), time.time() - t0))
@nb.njit(cache=True)
def ls(V0, reg, X, W, Xl, own, Vg_n, nd, st, rst, ee, ww, forb_l, goal_l, BIG, DTN, WTHR, pe):
    # WTHR здесь — порог веса принятых вершин ребра для LS
    E, K = X.shape; nreg = len(reg)
    s = np.zeros(E); sm = np.zeros(E); mx = np.full(E, -1e30)
    for e in range(E):                                        # вклад вершин вне области (граница = V0)
        for j in range(K):
            if Xl[e, j] < 0:
                v = V0[X[e, j]]; w = W[e, j]
                if v < BIG / 2 and w > 0:
                    s[e] += w * v; sm[e] += w
                    if v > mx[e]: mx[e] = v
    tent = np.full(nreg, BIG); acc = np.zeros(nreg, np.bool_)
    for k in range(len(nd)):
        i = nd[k]; t = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            if sm[e] >= WTHR:
                val = DTN + s[e] + (1 - sm[e]) * (mx[e] + pe)
                if val < t: t = val
        tent[i] = t
    h = [(0.0, np.int64(0))]; h.pop()
    for i in range(nreg):
        if goal_l[i]: tent[i] = 0.
        if not forb_l[i] and tent[i] < BIG / 2: h.append((tent[i], np.int64(i)))
    heapq.heapify(h); npop = 0; nupd = 0
    while len(h) > 0:
        t, i = heapq.heappop(h)
        if acc[i] or t > tent[i]: continue
        acc[i] = True; npop += 1
        for r in range(rst[i], rst[i + 1]):
            e = ee[r]; w = ww[r]; s[e] += w * t; sm[e] += w
            if t > mx[e]: mx[e] = t
            n = own[e]
            if acc[n] or forb_l[n] or goal_l[n] or sm[e] < WTHR: continue
            val = DTN + s[e] + (1 - sm[e]) * (mx[e] + pe); nupd += 1
            if val < tent[n]:
                tent[n] = val; heapq.heappush(h, (val, n))
    Vr = np.where(acc | goal_l, tent, BIG)
    return Vr, npop, nupd
forb_l = forb_g[reg]
Vg_ = V_gs[reg]; fa = Vg_ < BIG / 2; ndi = nd.astype(np.int64); sti = st.astype(np.int64)
fx = forb_g.copy(); fx[reg[goal_l]] = True
def stats(Vr):
    fb = Vr < BIG / 2; d = (Vr - Vg_)[fa & fb]
    return 'BIGрасх %d | мед %+.4f p90 %+.4f p99 %+.4f min %+.3f max %+.3f | |d|>.01 %.3f >.1 %.3f' % ((fa != fb).sum(), np.median(d), np.percentile(d, 90), np.percentile(d, 99), d.min(), d.max(), (np.abs(d) > .01).mean(), (np.abs(d) > .1).mean())
ls(V0, reg, X, W, Xl, own, Vg_n, ndi, sti, rst, ee, ww, forb_l, goal_l, BIG, DTN, WTHR, 1.)
# ===== research-31: область = замыкание вверх по argmin-политике V0 от запрещённых узлов =====
# argmin-ребро каждого узла по V0 (как в solve: val = DTN + s + (1-sm)(mx+pe), все вершины с V0<BIG/2)
@nb.njit(cache=True)
def argmin_edge(V0, X, W, st, Vg_n, BIG, DTN, WTHR, pe):
    G = len(st) - 1; am = np.full(G, -1, np.int64)
    for k in range(G):
        best = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            sm = 0.; s = 0.; mx = -1e30
            for j in range(W.shape[1]):
                w = W[e, j]; v = V0[X[e, j]]
                if v < BIG / 2:
                    s += w * v; sm += w
                    if v > mx: mx = v
            if sm < WTHR: continue
            val = DTN + s + (1 - sm) * (mx + pe)
            if val < best: best = val; am[k] = e
    return am
t0 = time.time(); am = argmin_edge(V0, X, W, sti, Vg_n, BIG, DTN, WTHR, 1.); am = argmin_edge(V0, X, W, sti, Vg_n, BIG, DTN, WTHR, 1.); t_am = time.time() - t0
print('argmin-рёбра: %.2f с; узлов с целью напрямую (am=-1): %d из %d' % (t_am, (am < 0).sum(), len(am)))
# точность утверждения: у узлов, НЕ затронутых, V_gs == V0?
@nb.njit(cache=True)
def closure(am, X, W, Xl, nd, nreg, forb_l, w0, rpst, rpk):
    aff = forb_l.copy(); q = np.flatnonzero(forb_l); qq = list(q); head = 0
    while head < len(qq):
        v = qq[head]; head += 1
        for r in range(rpst[v], rpst[v + 1]):
            k = rpk[r]; i = nd[k]
            if aff[i]: continue
            aff[i] = True; qq.append(i)
    return aff
nG = len(am); A_e = am[am >= 0]; A_k = np.flatnonzero(am >= 0)
Xa = Xl[A_e]; Wa = W[A_e]
for w0 in (0., .02, .1):
    m = (Xa >= 0) & (Wa > max(w0, 1e-12)); kk, jj = np.nonzero(m); vv = Xa[kk, jj]; o = np.argsort(vv, kind='stable')
    rpk = A_k[kk[o]].astype(np.int64); rpst = np.searchsorted(vv[o], np.arange(nreg + 1)).astype(np.int64)
    t0 = time.time(); aff = closure(am, X, W, Xl, ndi, nreg, forb_l, w0, rpst, rpk); tc = time.time() - t0
    un = ~aff; dv = np.abs(V_gs[reg][un] - V0[reg][un]); dv = dv[np.isfinite(dv) & (V_gs[reg][un] < BIG / 2)]
    nb_un_big = int(((V_gs[reg][un] >= BIG / 2) & (V0[reg][un] < BIG / 2)).sum())
    print('w0 %.2f: затронуто %d (%.3f области; запрещено %d), замыкание %.2f с | вне затронутых: |V_gs−V0| max %.4f p99 %.5f, стало BIG %d' % (w0, aff.sum(), aff.mean(), forb_l.sum(), tc, dv.max() if len(dv) else 0, np.percentile(dv, 99) if len(dv) else 0, nb_un_big))
    # LS только по затронутым незапрещённым: подобласть
    sub = np.flatnonzero(aff); reg2 = reg[sub]; loc2 = np.full(len(P), -1, np.int64); loc2[reg2] = np.arange(len(reg2))
    gk = np.flatnonzero(aff[nd]); lens = np.diff(st)[gk]; st2 = np.r_[0, np.cumsum(lens)].astype(np.int64)
    eidx = np.concatenate([np.arange(st[k], st[k + 1]) for k in gk]) if len(gk) else np.zeros(0, np.int64)
    X2 = X[eidx]; W2 = W[eidx]; nd2 = loc2[reg[nd[gk]]].astype(np.int64); Vg2 = Vg_n[gk]
    t0 = time.time(); Xl2 = loc2[X2]; own2 = np.repeat(nd2, lens).astype(np.int64); m2 = (Xl2 >= 0) & (W2 > 1e-12)
    e2, j2 = np.nonzero(m2); key2 = Xl2[e2, j2]; o2 = np.argsort(key2, kind='stable'); ee2 = e2[o2].astype(np.int64); ww2 = W2[e2, j2[o2]].astype(np.float64)
    rst2 = np.searchsorted(key2[o2], np.arange(len(reg2) + 1)); tb = time.time() - t0
    t0 = time.time(); Vr2, np2_, nu2 = ls(V0, reg2, X2, W2, Xl2, own2, Vg2, nd2, st2, rst2, ee2, ww2, forb_g[reg2], A.goal[reg2], BIG, DTN, WTHR, .3); tl = time.time() - t0
    V = V0.copy(); V[reg2] = Vr2; t1 = time.time()
    s1 = stats(V[reg]); order = np.argsort(V[reg[nd]], kind='stable'); sweep(V, reg[nd], st, X, W, Vg_n, order, BIG, DTN, WTHR, fx, 1.)
    A.V = V0.copy(); A.V[reg2] = Vr2; vl = float(A.vstar(y[None])[0])
    print('   LS по затронутым: узлов %d, рёбер %d, вхождений %d; индекс %.2f с, LS %.3f с | %s | +1 GS по области %s | V(аг) %.3f (GS %.3f)' % (len(reg2), len(eidx), len(ee2), tb, tl, s1, stats(V[reg]), vl, vg))
