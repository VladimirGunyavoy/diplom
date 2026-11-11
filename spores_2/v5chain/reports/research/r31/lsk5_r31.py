# research-31: ускорение LS (Б-LS) локальностью памяти: перенумерация по V0 + упаковка сумм рёбер (AoS float32). Атлас DTN .2 (aida 1).
import os, sys, pickle, time
os.environ.update(SYS='car', M='3', SIDE='0', GM='.25', GLIM='0', FRAC='1', PESS='1', GS='300', OMP_NUM_THREADS='1')
os.environ.setdefault('DTN', '.2')
sys.path.insert(0, '.'); sys.path.insert(0, '../v7/src/cells7')
import numpy as np, envs_dyn as D, growCar as G, __main__, inspect, numba as nb, heapq
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(sys.argv[1]), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
P = A.P; V0 = A.V.copy(); BIG = G.BIG; DTN = G.DTN; US = G.US; WTHR = G.WTHR; kap = np.sqrt(-2*np.log(.03)); T_PR = float(os.environ.get('TPR', 2.))
print('атлас: узлов %d, DTN %.2f' % (len(P), DTN))
rng = np.random.default_rng(0); dC = np.linalg.norm(P[:, :2] - D.C0[0], axis=1)
cand = np.flatnonzero((V0 > 3.0) & (V0 < 3.4) & (dC > 1.0) & (dC < 2.0)); ia = rng.choice(cand); y = P[ia]; va = V0[ia]
reg = np.flatnonzero((V0 <= va) & (va - V0 <= T_PR)); print('агент', y, 'V0 %.3f; оболочка %d узлов' % (va, len(reg)))
t0 = time.time(); Is, Xs, Ws, Vg = [], [], [], np.full(len(reg), np.inf)
for u in US:
    Vg = np.minimum(Vg, A.tgoal(P[reg], u)); a, b, c = A.stencils(G.step(P[reg], u)); Is.append(a.astype(np.int32)); Xs.append(b.astype(np.int32)); Ws.append(c.astype(np.float32)); del a, b, c
I = np.concatenate(Is); del Is; X = np.concatenate(Xs); del Xs; W = np.concatenate(Ws); del Ws
o = np.argsort(I, kind='stable'); I, X, W = I[o], X[o], W[o]; del o
st = np.r_[0, np.flatnonzero(I[1:] != I[:-1]) + 1, len(I)].astype(np.int64); nd = I[st[:-1]].astype(np.int64); del I
print('рёбер %d, стенсилы %.1f с' % (len(X), time.time() - t0))
tau = np.maximum(va - V0[reg], 0.); sg = np.minimum(np.sqrt(.01 * tau ** 3 / 3), .25)
fm = (np.linalg.norm(P[reg][:, None, :2] - D.C0[None], axis=2) - D.HR - .1 - kap * sg[:, None]).min(1) < 0
goal_l = A.goal[reg]; Vg_n = np.where(np.isfinite(Vg[nd]), Vg[nd], BIG); print('запрещено %d' % fm.sum())
nreg = len(reg)
def prep(nd, st, X, W, perm=None):
    loc = np.full(len(P), -1, np.int64); loc[reg] = np.arange(nreg) if perm is None else perm
    Xl = loc[X]; own = np.repeat(nd, np.diff(st)).astype(np.int64)
    m = (Xl >= 0) & (W > 1e-12); ee, jj = np.nonzero(m); key = Xl[ee, jj]; o = np.argsort(key, kind='stable')
    ee2 = ee[o].astype(np.int64); ww = W[ee2, jj[o]].astype(np.float64); rst = np.searchsorted(key[o], np.arange(nreg + 1))
    return Xl, own, ee2, ww, rst
@nb.njit(cache=True)
def ls(V0, X, W, Xl, own, Vg_n, nd, st, rst, ee, ww, forb_l, goal_l, BIG, DTN, WTHR, pe):
    E, K = X.shape; nreg = len(forb_l)
    s = np.zeros(E); sm = np.zeros(E); mx = np.full(E, -1e30)
    for e in range(E):
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
    heapq.heapify(h)
    while len(h) > 0:
        t, i = heapq.heappop(h)
        if acc[i] or t > tent[i]: continue
        acc[i] = True
        for r in range(rst[i], rst[i + 1]):
            e = ee[r]; w = ww[r]; s[e] += w * t; sm[e] += w
            if t > mx[e]: mx[e] = t
            n = own[e]
            if acc[n] or forb_l[n] or goal_l[n] or sm[e] < WTHR: continue
            val = DTN + s[e] + (1 - sm[e]) * (mx[e] + pe)
            if val < tent[n]:
                tent[n] = val; heapq.heappush(h, (val, n))
    return np.where(acc | goal_l, tent, BIG)
@nb.njit(cache=True)
def ls2(V0, X, W, Xl, own, Vg_n, nd, st, rst, ee, ww, forb_l, goal_l, BIG, DTN, WTHR, pe):
    # то же, суммы ребра в одной строке S[e] = (s, sm, mx) float32 — одна кэш-линия на обновление
    E, K = X.shape; nreg = len(forb_l)
    S = np.zeros((E, 3), np.float32); S[:, 2] = -1e30
    for e in range(E):
        for j in range(K):
            if Xl[e, j] < 0:
                v = V0[X[e, j]]; w = W[e, j]
                if v < BIG / 2 and w > 0:
                    S[e, 0] += w * v; S[e, 1] += w
                    if v > S[e, 2]: S[e, 2] = v
    tent = np.full(nreg, BIG); acc = np.zeros(nreg, np.bool_)
    for k in range(len(nd)):
        i = nd[k]; t = Vg_n[k]
        for e in range(st[k], st[k + 1]):
            if S[e, 1] >= WTHR:
                val = DTN + S[e, 0] + (1 - S[e, 1]) * (S[e, 2] + pe)
                if val < t: t = val
        tent[i] = t
    h = [(0.0, np.int64(0))]; h.pop()
    for i in range(nreg):
        if goal_l[i]: tent[i] = 0.
        if not forb_l[i] and tent[i] < BIG / 2: h.append((tent[i], np.int64(i)))
    heapq.heapify(h)
    while len(h) > 0:
        t, i = heapq.heappop(h)
        if acc[i] or t > tent[i]: continue
        acc[i] = True
        for r in range(rst[i], rst[i + 1]):
            e = ee[r]; w = ww[r]; S[e, 0] += w * t; S[e, 1] += w
            if t > S[e, 2]: S[e, 2] = t
            n = own[e]
            if acc[n] or forb_l[n] or goal_l[n] or S[e, 1] < WTHR: continue
            val = DTN + S[e, 0] + (1 - S[e, 1]) * (S[e, 2] + pe)
            if val < tent[n]:
                tent[n] = val; heapq.heappush(h, (val, n))
    return np.where(acc | goal_l, tent, BIG)
args = lambda Xl, own, ee, ww, rst, nd_, st_, X_, W_, Vgn, fl, gl: (V0, X_, W_, Xl, own, Vgn, nd_, st_, rst, ee, ww, fl, gl, BIG, DTN, WTHR, .3)
# (A) исходный порядок
t0 = time.time(); pa = prep(nd, st, X, W); tpa = time.time() - t0
ls(*args(*pa, nd, st, X, W, Vg_n, fm, goal_l)); t0 = time.time(); Va = ls(*args(*pa, nd, st, X, W, Vg_n, fm, goal_l)); ta = time.time() - t0
print('(A) исходный порядок: индекс %.2f с, LS %.3f с (%.2f мкс/узел)' % (tpa, ta, ta / nreg * 1e6)); del pa
# (B) перенумерация узлов по V0, рёбра — в порядке новых номеров владельцев
t0 = time.time(); rank = np.empty(nreg, np.int64); rank[np.argsort(V0[reg], kind='stable')] = np.arange(nreg)
gord = np.argsort(rank[nd], kind='stable'); lens = np.diff(st)[gord]; stB = np.r_[0, np.cumsum(lens)].astype(np.int64)
eidx = np.repeat(st[:-1][gord] - stB[:-1], lens) + np.arange(stB[-1]); XB = X[eidx]; WB = W[eidx]; ndB = rank[nd[gord]]; del eidx
fB = np.zeros(nreg, bool); fB[rank] = fm; gB = np.zeros(nreg, bool); gB[rank] = goal_l
pb = prep(ndB, stB, XB, WB, perm=rank); tpb = time.time() - t0
ls(*args(*pb, ndB, stB, XB, WB, Vg_n[gord], fB, gB)); t0 = time.time(); Vb = ls(*args(*pb, ndB, stB, XB, WB, Vg_n[gord], fB, gB)); tb = time.time() - t0
Vb_o = Vb[rank]; f = Va < BIG / 2
print('(B) перенумерация по V0: подготовка %.2f с, LS %.3f с (%.2f мкс/узел); max|ΔV| к (A) %.2e, BIG расх %d' % (tpb, tb, tb / nreg * 1e6, np.abs(Vb_o[f] - Va[f]).max(), ((Vb_o < BIG / 2) != f).sum()))
ls2(*args(*pb, ndB, stB, XB, WB, Vg_n[gord], fB, gB)); t0 = time.time(); Vc = ls2(*args(*pb, ndB, stB, XB, WB, Vg_n[gord], fB, gB)); tc = time.time() - t0
Vc_o = Vc[rank]
print('(C) + AoS float32: LS %.3f с (%.2f мкс/узел); max|ΔV| к (A) %.2e, p99 %.2e, BIG расх %d' % (tc, tc / nreg * 1e6, np.abs(Vc_o[f] - Va[f]).max(), np.percentile(np.abs(Vc_o[f] - Va[f]), 99), ((Vc_o < BIG / 2) != f).sum()))
