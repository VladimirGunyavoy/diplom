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
CF = os.path.expanduser('~/spore_v5/r31/shell_d2s1.npz')
if os.path.exists(CF):
    z = np.load(CF); X, W, st, nd, Vg = z['X'], z['W'], z['st'], z['nd'], z['Vg']; print('стенсилы из кэша, рёбер %d' % len(X))
else:
    t0 = time.time(); Is, Xs, Ws, Vg = [], [], [], np.full(len(reg), np.inf)
    for u in US:
        Vg = np.minimum(Vg, A.tgoal(P[reg], u)); a, b, c = A.stencils(G.step(P[reg], u)); Is.append(a.astype(np.int32)); Xs.append(b.astype(np.int32)); Ws.append(c.astype(np.float32)); del a, b, c
    I = np.concatenate(Is); del Is; X = np.concatenate(Xs); del Xs; W = np.concatenate(Ws); del Ws
    o = np.argsort(I, kind='stable'); I, X, W = I[o], X[o], W[o]; del o
    st = np.r_[0, np.flatnonzero(I[1:] != I[:-1]) + 1, len(I)].astype(np.int64); nd = I[st[:-1]].astype(np.int64); del I
    print('рёбер %d, стенсилы %.1f с' % (len(X), time.time() - t0))
    np.savez(CF, X=X, W=W, st=st, nd=nd, Vg=Vg)
tau = np.maximum(va - V0[reg], 0.); sg = np.minimum(np.sqrt(.01 * tau ** 3 / 3), .25)
goal_l = A.goal[reg]; Vg_n = np.where(np.isfinite(Vg[nd]), Vg[nd], BIG); nreg = len(reg)
H = D.Hazards('fixed', 0); Preg = P[reg][:, :2]
def forb(t):   # облако по ИСТИННОМУ закону: центр диска в момент t + τ(узла), раздутие κσ(τ)
    c = H.pos(t + tau)             # (n, 3, 2)
    return (np.linalg.norm(Preg[:, None, :] - c, axis=2) - D.HR - .1 - kap * sg[:, None]).min(1) < 0
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
pa = prep(nd, st, X, W)
run = lambda fm: ls2(V0, X, W, pa[0], pa[1], Vg_n, nd, st, pa[4], pa[2], pa[3], fm, goal_l, BIG, DTN, WTHR, .3)
f0 = forb(0.); run(f0); t0 = time.time(); Vb = run(f0); print('LS (AoS f32) %.2f с, запрещено %d' % (time.time() - t0, f0.sum()))
fin = Vb < BIG / 2
for k in (6, 12, 24, 36):
    dt = k * D.DT; fk = forb(dt); Vk = run(fk); fk_ = Vk < BIG / 2; both = fin & fk_
    dv = np.abs(Vk[both] - Vb[both]); chf = (f0 != fk).sum()
    print('k %2d (%.2f с): смена запрета %d узлов (%.3f) | конечн.→BIG %d, BIG→конечн. %d | |ΔV| >.01: %.3f  >.05: %.3f  >.1: %.3f  мед %.4f p90 %.4f' % (k, dt, chf, chf / nreg, (fin & ~fk_).sum(), (~fin & fk_).sum(), (dv > .01).mean(), (dv > .05).mean(), (dv > .1).mean(), np.median(dv), np.percentile(dv, 90)))
