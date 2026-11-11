"""b24 ДИНАМИК-КАР (см. хвост): основа car_val.py — СТАТИК-КАР, рецепт круга: обратное дерево v6 (SysN, θ периодический) на 4D (x,y,θ,v); V=время-до-цели итерацией ценностей по графу узлов
(W_i=min_a[τ+Ŵ(flow(i,a,τ))], Ŵ(q)=min_{K ближайших b}(W_b+KV·|q-b|_scaled), диски/стены/|v| — запрещённые рёбра); агент жадный по τ+Ŵ + фильтр безопасности
(зазор до дисков по пути шага ≥ FM). Модель — ref_car100 (K .625, a±1.5, ω±1.8, v∈[-1,2], 3 диска r.7, цель (4,3.6) r.45, стены ±5).
Запуск (aida, рядом с b22/): runpc.sh car_val.py NB K mode(sel|test) TREESEED [T1]; ENV: KV (1.0), RB (.3), TAU (.2), FM (.1), MARG (.12), SC (.25,.25,.4,.3)
sel: старты seed 2000 (первые NSEL=32), эталон b24/ref*/car_q<i>.npy; test: starts100.npy (seed 1000), T_ref из r20 json (TREF=путь)."""
import os, sys, time, json, glob
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'v6'))
import numpy as np, numba as nb
from src.atlas6.adaptive_nd import SysN, build_back
NB, KN, MODE, TS, K0, K1 = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, os.path.join(HERE, '..')); import envs_dyn as D
KV = float(os.environ.get('KV', 1.0)); RB = float(os.environ.get('RB', .3)); TAU = float(os.environ.get('TAU', .2)); FM = float(os.environ.get('FM', .1)); MARG = float(os.environ.get('MARG', .12)); HB = float(os.environ.get('HB', 1.0))
SC = tuple(float(v) for v in os.environ.get('SC', '.25,.25,.4,.3').split(','))
K, AM, WM, VL, VH, XL = .625, 1.5, 1.8, -1., 2., 5.
HZ = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); HR = .70; GOAL = np.array([4., 3.6]); GR = .45
ACT = np.array([[a, w] for a in (-AM, 0., AM) for w in (-WM, 0., WM)])

@nb.njit(inline='always')
def _f(x, y, th, v, a, w): return v * np.cos(th), v * np.sin(th), w, a - K * v
@nb.njit
def rk(x, y, th, v, a, w, h, n):
    for _ in range(n):
        k1 = _f(x, y, th, v, a, w); k2 = _f(x + h / 2 * k1[0], y + h / 2 * k1[1], th + h / 2 * k1[2], v + h / 2 * k1[3], a, w)
        k3 = _f(x + h / 2 * k2[0], y + h / 2 * k2[1], th + h / 2 * k2[2], v + h / 2 * k2[3], a, w); k4 = _f(x + h * k3[0], y + h * k3[1], th + h * k3[2], v + h * k3[3], a, w)
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); th += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]); v += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    return x, y, th, v
@nb.njit
def clear(x, y):
    return XL - max(abs(x), abs(y))   # carfree: только стены (диски — пометка узлов)
@nb.njit(parallel=True)
def flowb(P, a, w, t):
    out = np.empty_like(P); n = max(1, int(np.ceil(abs(t) / .02)))
    for i in nb.prange(P.shape[0]):
        x, y, th, v = rk(P[i, 0], P[i, 1], P[i, 2], P[i, 3], a, w, t / n, n); out[i, 0] = x; out[i, 1] = y; out[i, 2] = th; out[i, 3] = v
    return out
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); u = ACT[s]
    return flowb(P.reshape(-1, 4).copy(), u[0], u[1], float(t)).reshape(P.shape)
def in_goal(P): P = np.asarray(P); return np.hypot(P[..., 0] - GOAL[0], P[..., 1] - GOAL[1]) <= GR - .05
def blocked(P):
    P = np.asarray(P); m = XL - np.maximum(np.abs(P[..., 0]), np.abs(P[..., 1]))
    return m < MARG
def okv(P): P = np.asarray(P); return (P[..., 3] >= VL) & (P[..., 3] <= VH)
def mkS(seed):
    rng = np.random.default_rng(seed); n = 300; r = .35 * np.sqrt(rng.uniform(size=n)); a = rng.uniform(0, 2 * np.pi, n)
    seeds = np.c_[GOAL[0] + r * np.cos(a), GOAL[1] + r * np.sin(a), rng.uniform(-np.pi, np.pi, n), rng.uniform(VL, VH, n)]
    return SysN(flow, 9, SC, in_goal, seeds, per=(0, 0, 2 * np.pi, 0), ok=okv, blocked=blocked)

@nb.njit(parallel=True)
def edges(P, tau):
    n = P.shape[0]; E = np.zeros((n, 9, 4)); OK = np.zeros((n, 9), np.bool_); GL = np.zeros((n, 9), np.bool_)
    for i in nb.prange(n):
        for s in range(9):
            x, y, th, v = P[i, 0], P[i, 1], P[i, 2], P[i, 3]; ok = True; h = tau / 10
            for _ in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1)
                if clear(x, y) < MARG or v < VL or v > VH: ok = False
                if (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= (GR - .05) ** 2: GL[i, s] = True; break
            E[i, s, 0] = x; E[i, s, 1] = y; E[i, s, 2] = th; E[i, s, 3] = v; OK[i, s] = ok
    return E, OK, GL
@nb.njit(parallel=True)
def vit(J, D, OK, GL, ING, nP, tau, kv, nit, big):
    W = np.full(nP, big); K_ = J.shape[1]
    for i in range(nP):
        if ING[i]: W[i] = 0.
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            best = big
            for a in range(9):
                if not OK[i, a]: continue
                if GL[i, a]: c = tau
                else:
                    m = 1e30
                    for k in range(K_):
                        v = W[J[i * 9 + a, k]] + kv * D[i * 9 + a, k]
                        if v < m: m = v
                    c = tau + m
                if c < best: best = c
            Wn[i] = best
        d = np.max(np.abs(Wn - W)); W = Wn
        if d < 1e-4: break
    return W, it, d


HX = float(os.environ.get('HOLDX', 0)); HYST = float(os.environ.get('HYST', 0)); HH = float(os.environ.get('HH', 2.)); EPS = float(os.environ.get('EPS', .03)); KAPC = float(np.sqrt(-2 * np.log(EPS))); QC = float(os.environ.get('QC', .01)); SMAX = float(os.environ.get('SMAX', .5)); MGN = float(os.environ.get('MG', .12)); NITD = int(os.environ.get('NITD', 40)); BIG = 1e3
@nb.njit(parallel=True)
def vit2(J, D, OK, GL, ING, FLAG, W0, nP, tau, kv, nit, big, tol):
    W = W0.copy(); K_ = J.shape[1]
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            if FLAG[i]: Wn[i] = big; continue
            best = big
            for a in range(9):
                if not OK[i, a]: continue
                if GL[i, a]: c = tau
                else:
                    m = 1e30
                    for k in range(K_):
                        v = W[J[i * 9 + a, k]] + kv * D[i * 9 + a, k]
                        if v < m: m = v
                    c = tau + m
                if c < best: best = c
            Wn[i] = best
        d = np.max(np.abs(Wn - W)); W = Wn
        if d < tol: break
    return W, it
@nb.njit
def cand(x, y, th, v, a, w, P0, Wv, kap, q, smax, tau, hb):
    """шаг .2 с + торможение hb с под ПРОГНОЗ дисков (центр p0+w t, облако κσ(t), σ=min(√(qt³/3), smax)); (состояние, мин. зазор, вход в цель или -1)"""
    mn = 9.; hit = -1.; t = 0.; h = .02
    for k in range(int(round(tau / h)) + int(hb / .05)):
        if k < int(round(tau / h)): x, y, th, v = rk(x, y, th, v, a, w, h, 1); t += h
        else: x, y, th, v = rk(x, y, th, v, -AM if v > 0 else AM, 0., .05, 1); t += .05
        sg = min(np.sqrt(q * t ** 3 / 3.), smax); mn = min(mn, XL - max(abs(x), abs(y)))
        for i in range(3):
            cx = min(max(P0[i, 0] + Wv[i, 0] * t, -(XL - HR)), XL - HR); cy = min(max(P0[i, 1] + Wv[i, 1] * t, -(XL - HR)), XL - HR)
            mn = min(mn, np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - HR - kap * sg)
        if k < int(round(tau / h)) and hit < 0 and (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= (GR - .05) ** 2: hit = t; break
    return x, y, th, v, mn, hit
@nb.njit
def execute_dyn(x, y, th, v, a, w, dt, n, PT):
    """шаг по ИСТИННЫМ дискам PT (n, 3, 2): (состояние, мин. истинный зазор до дисков/стен, вход в цель)"""
    mn = 9.; hit = -1.
    for k in range(n):
        x, y, th, v = rk(x, y, th, v, a, w, dt, 1); mn = min(mn, XL - max(abs(x), abs(y)))
        for i in range(3): mn = min(mn, np.sqrt((x - PT[k, i, 0]) ** 2 + (y - PT[k, i, 1]) ** 2) - HR)
        if hit < 0 and (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= GR ** 2: hit = (k + 1) * dt; break
    return x, y, th, v, mn, hit

import math
LIMV = float(os.environ.get('LIMV', 1e9)); TOLV = float(os.environ.get('TOLV', 2e-3)); WARM = int(os.environ.get('WARM', 0)); GPU = int(os.environ.get('GPUVI', 0)); KG = int(os.environ.get('KG', 24))
if GPU:
    from numba import cuda
    @cuda.jit
    def k_vit(J, D, OK, GL, ING, FLAG, W, Wn, nP, tau, kv, big, W0, lim):
        i = cuda.grid(1)
        if i >= nP: return
        if W0[i] > lim: Wn[i] = W[i]; return   # окно: узлы дальше от цели, чем агент+LIMV, не пересчитываются
        if ING[i]: Wn[i] = 0.; return
        if FLAG[i]: Wn[i] = big; return
        best = big
        for a in range(9):
            if not OK[a, i]: continue
            if GL[a, i]: c = tau
            else:
                m = 1e30
                for k in range(J.shape[1]):
                    v = W[J[a, k, i]] + kv * D[a, k, i]
                    if v < m: m = v
                c = tau + m
            if c < best: best = c
        Wn[i] = best
    @cuda.jit
    def k_flag(X, Y, W0, FL, nP, W0x, P0, Wv, hh, qc, smax, kap, rr):
        i = cuda.grid(1)
        if i >= nP: return
        tb = W0x - W0[i]
        if tb < 0.: tb = 0.
        if tb > hh: tb = hh
        sg = math.sqrt(qc * tb * tb * tb / 3.)
        if sg > smax: sg = smax
        f = False
        for dd in range(3):
            mx = P0[dd, 0] + Wv[dd, 0] * tb; my = P0[dd, 1] + Wv[dd, 1] * tb
            if math.sqrt((X[i] - mx) ** 2 + (Y[i] - my) ** 2) < rr + kap * sg: f = True
        FL[i] = f
    def vit_gpu(ag, FL, nit, tol):
        dF = FL; a = (ag.dWprev if WARM else cuda.to_device(ag.W0f)); b = cuda.device_array_like(a); tpb = 128; nbk = (ag.nP + tpb - 1) // tpb
        for it in range(nit):
            k_vit[nbk, tpb](ag.dJ, ag.dD, ag.dOK, ag.dGL, ag.dING, dF, a, b, ag.nP, np.float32(TAU), np.float32(KV), np.float32(BIG), ag.dW0, np.float32(ag.lim)); a, b = b, a
            if it % 4 == 3 and np.abs(a.copy_to_host() - b.copy_to_host()).max() < tol: break
        ag.dWprev = a; return a.copy_to_host(), it
TIE = float(os.environ.get('TIE', 0)); DEP = int(os.environ.get('DEP', 2)); ESCN = int(os.environ.get('ESCN', 0)); ESCD = int(os.environ.get('ESCD', 7)); ESCB = int(os.environ.get('ESCB', 20000))
@nb.njit
def expand(X, tau):
    n = X.shape[0]; out = np.empty((n * 9, 4)); okk = np.ones(n * 9, np.bool_)
    for i in range(n):
        for s in range(9):
            x, y, th, v = X[i, 0], X[i, 1], X[i, 2], X[i, 3]; h = tau / 10
            for _ in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1)
                if XL - max(abs(x), abs(y)) < MARG or v < VL or v > VH: okk[i * 9 + s] = False
            out[i * 9 + s, 0] = x; out[i * 9 + s, 1] = y; out[i * 9 + s, 2] = th; out[i * 9 + s, 3] = v
    return out, okk
def tie_wq(ag, Q, ok, Wq):
    """ТАЙ-БРЕЙК: разброс V̂ по допустимым действиям < TIE (гребень у покоя) → поиск вперёд DEP шагов (9^DEP листьев), V̂ листа; возвращает Wq по первому действию (листья дальше стены/|v| отброшены; безопасность — только на 1-м шаге)"""
    if TIE <= 0 or not ok.any(): return Wq
    w = Wq[ok]
    if w.max() - w.min() >= TIE: ag.streak = 0; return Wq
    ag.ntie = getattr(ag, 'ntie', 0) + 1; ag.streak = getattr(ag, 'streak', 0) + 1; cur = Q; first = np.arange(9); okc = ok.copy()
    if ESCN > 0 and ag.streak >= ESCN:   # b25: эскалация — луч шириной ESCB до глубины ESCD
        ag.nesc = getattr(ag, 'nesc', 0) + 1; Wl = None
        for lev in range(ESCD - 1):
            cur, okn = expand(np.ascontiguousarray(cur), TAU); first = np.repeat(first, 9); okc = np.repeat(okc, 9) & okn
            d, j = ag.tree.query(ag.S.wrap(cur) / ag.S.scale, k=min(KN * 3, ag.nP)); Wl = np.min(ag.W[j % ag.nP] + KV * d, axis=1); Wl[~okc] = 1e9
            if len(Wl) > ESCB and lev < ESCD - 2:
                keep = np.argpartition(Wl, ESCB)[:ESCB]; cur, first, okc, Wl = cur[keep], first[keep], okc[keep], Wl[keep]
        out = np.full(9, 1e9)
        for f in range(9):
            m = first == f
            if m.any(): out[f] = Wl[m].min()
        return out
    for _ in range(DEP - 1):
        cur, okn = expand(np.ascontiguousarray(cur), TAU); first = np.repeat(first, 9); okc = np.repeat(okc, 9) & okn
    d, j = ag.tree.query(ag.S.wrap(cur) / ag.S.scale, k=min(KN * 3, ag.nP)); Wl = np.min(ag.W[j % ag.nP] + KV * d, axis=1); Wl[~okc] = 1e9
    return Wl.reshape(9, -1).min(1)

class AgentDyn:
    """дерево carfree (без дисков) + W0 (VI один раз); на каждом шаге: пометка узлов облаком прогноза → VI с тёплым стартом → жадный выбор + фильтр по прогнозу. Закон дисков агенту неизвестен: только наблюдения p(t), p(t−.2)."""
    def __init__(self, S, back):
        global KV
        from scipy.spatial import cKDTree
        self.S, self.tree, self.P = S, back[5], np.asarray(back[0]); self.nP = len(self.P)
        E, self.OK, self.GL = edges(self.P, TAU); Ew = S.wrap(E.reshape(-1, 4)); d, j = self.tree.query(Ew / S.scale, k=min(KN * 3, len(self.tree.data)), workers=-1)
        self.J = (j % self.nP).astype(np.int32); self.D = d.astype(np.float64); self.ING = in_goal(self.P)
        self.W0, it, _ = vit(self.J, self.D, self.OK, self.GL, self.ING, self.nP, TAU, KV, 300, BIG); self.xy = cKDTree(self.P[:, :2]); self.nflag = 0; self.tvi = 0.; self.tfl = 0.
        if GPU:
            self.dJ = cuda.to_device(np.ascontiguousarray(self.J.reshape(self.nP, 9, -1)[:, :, :KG].transpose(1, 2, 0))); self.dD = cuda.to_device(np.ascontiguousarray(self.D.reshape(self.nP, 9, -1)[:, :, :KG].transpose(1, 2, 0).astype(np.float32))); self.dOK = cuda.to_device(np.ascontiguousarray(self.OK.T)); self.dGL = cuda.to_device(np.ascontiguousarray(self.GL.T)); self.dING = cuda.to_device(self.ING); self.W0f = self.W0.astype(np.float32); self.dW0 = cuda.to_device(self.W0f); self.dWprev = cuda.to_device(self.W0f); self.dX = cuda.to_device(np.ascontiguousarray(self.P[:, 0])); self.dY = cuda.to_device(np.ascontiguousarray(self.P[:, 1]))
        i0 = np.random.default_rng(0).choice(self.nP, min(self.nP, 300000), replace=False); jj = self.J[i0 * 9 + 4, 0]; dP = S.wrap(self.P[jj] - self.P[i0]) / S.scale; dd = np.linalg.norm(dP, axis=1); fin = (self.W0[i0] < 1e2) & (self.W0[jj] < 1e2) & (dd > .05)
        self.kv_auto = float(np.quantile(np.abs(self.W0[i0] - self.W0[jj])[fin] / dd[fin], .9)); print('  KV_auto (кв.9 |ΔW0|/d по рёбрам) %.3f' % self.kv_auto, flush=True)
        if os.environ.get('KVAUTO') == '1':   # b25: неподвижная точка KV = q.9 Липшица W0(KV)
            for _ in range(int(os.environ.get('KVIT', 8))):
                kn_ = min(self.kv_auto, float(os.environ.get('KVMAX', 1.0))); conv = abs(kn_ - KV) < .05 * kn_; KV = kn_
                self.W0, it, _r = vit(self.J, self.D, self.OK, self.GL, self.ING, self.nP, TAU, KV, 300, BIG)
                fin = (self.W0[i0] < 1e2) & (self.W0[jj] < 1e2) & (dd > .05); self.kv_auto = float(np.quantile(np.abs(self.W0[i0] - self.W0[jj])[fin] / dd[fin], .9)); print('  KVAUTO it: KV=%.3f -> kv_auto %.3f' % (KV, self.kv_auto), flush=True)
                if conv: break
        print('  carfree: узлов %d, W0 мед %.2f макс %.2f (it %d)' % (self.nP, np.median(self.W0), self.W0.max(), it), flush=True)
    def plan(self, x, P0, Wv):
        d, j = self.tree.query(self.S.wrap(x[None]) / self.S.scale, k=min(KN * 3, self.nP)); W0x = float(np.min(self.W0[j % self.nP] + KV * d)); self.lim = W0x + LIMV; self.W0x = W0x
        tf = time.perf_counter()
        if GPU:
            dF = cuda.device_array(self.nP, np.bool_); tpb = 128; k_flag[(self.nP + tpb - 1) // tpb, tpb](self.dX, self.dY, self.dW0, dF, self.nP, np.float64(W0x), np.ascontiguousarray(P0), np.ascontiguousarray(Wv), np.float64(HH), np.float64(QC), np.float64(SMAX), np.float64(KAPC), np.float64(HR + MGN))
            t0 = time.perf_counter(); self.tfl = t0 - tf; self.W, self.nit = vit_gpu(self, dF, NITD, TOLV); self.W = self.W.astype(np.float64); self.tvi = time.perf_counter() - t0; self.nflag = -1; return
        FL = np.zeros(self.nP, bool)
        for i in range(3):
            idx = np.array(self.xy.query_ball_point(P0[i], 1.0 + HR + MGN + KAPC * SMAX + np.hypot(*Wv[i]) * HH), int)
            if len(idx) == 0: continue
            tb = np.clip(W0x - self.W0[idx], 0., HH); sg = np.minimum(np.sqrt(QC * tb ** 3 / 3.), SMAX); m = P0[i][None] + Wv[i][None] * tb[:, None]
            FL[idx[np.hypot(self.P[idx, 0] - m[:, 0], self.P[idx, 1] - m[:, 1]) < HR + KAPC * sg + MGN]] = True
        t0 = time.perf_counter(); self.tfl = t0 - tf
        if GPU: self.W, self.nit = vit_gpu(self, FL, NITD, TOLV); self.W = self.W.astype(np.float64)
        else: self.W, self.nit = vit2(self.J, self.D, self.OK, self.GL, self.ING, FL, self.W0, self.nP, TAU, KV, NITD, BIG, 2e-3)
        self.tvi = time.perf_counter() - t0; self.nflag = int(FL.sum())
    def act(self, x, P0, Wv):
        self.plan(x, P0, Wv); sc = np.full(9, 1e9); cl = np.zeros(9); Q = np.zeros((9, 4))
        for s in range(9):
            xe, ye, the, ve, mn, hit = cand(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], P0, Wv, KAPC, QC, SMAX, TAU, HB); cl[s] = mn; Q[s] = (xe, ye, the, ve)
            if hit >= 0 and mn >= FM: sc[s] = hit
        ok = (cl >= FM) & (Q[:, 3] >= VL) & (Q[:, 3] <= VH)
        d0, j0 = self.tree.query(self.S.wrap(x[None]) / self.S.scale, k=min(KN * 3, self.nP)); self.Wx = float(np.min(self.W[j0 % self.nP] + KV * d0))
        if (sc < 1e9).any(): return int(np.argmin(sc))
        if not ok.any(): return int(np.argmax(cl))
        d, j = self.tree.query(self.S.wrap(Q) / self.S.scale, k=min(KN * 3, self.nP)); Wq = np.min(self.W[j % self.nP] + KV * d, axis=1)
        if HX > 0:   # HOLDX: то же действие удерживается HX с — на гребне V (v≈0, Ŵ плоский) шаг .2 с не различает действия; цена = длительность + Ŵ(конец); берём лучшее из двух горизонтов
            Q2 = np.zeros((9, 4)); c2 = np.zeros(9)
            for s in range(9):
                xe, ye, the, ve, mn, hit = cand(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], P0, Wv, KAPC, QC, SMAX, HX, HB); Q2[s] = (xe, ye, the, ve); c2[s] = mn
            ok2 = (c2 >= FM) & (Q2[:, 3] >= VL) & (Q2[:, 3] <= VH); d2, j2 = self.tree.query(self.S.wrap(Q2) / self.S.scale, k=min(KN * 3, self.nP)); Wq2 = np.min(self.W[j2 % self.nP] + KV * d2, axis=1)
            Wq = np.where(ok2 & (HX + Wq2 < TAU + Wq), HX + Wq2 - TAU, Wq)
        Wq = tie_wq(self, Q, ok, Wq); self.Wq, self.cl, self.okc = Wq, cl, ok
        return self.hyst(np.where(ok, Wq, 1e9), ok)
    def hyst(self, c, ok):
        k = int(np.argmin(c)); pv = getattr(self, 'prev', None)
        if HYST > 0 and pv is not None and ok[pv] and c[pv] <= c[k] + HYST: k = pv
        self.prev = k; return k
def episode(ag, x0, H, tmax=60.):
    ag.prev = None; x = np.array(x0, float); traj = [x.copy()]; ag.last_traj = traj; t = 0.; mn = 9.; tms = []; tv = []; Pp = H.pos(0.); nits = []; tfl = []; nsub = int(round(TAU / .002))
    while t < tmax:
        P0 = H.pos(t); Wv = (P0 - Pp) / TAU if t > 0 else np.zeros((3, 2)); Pp = P0
        t1 = time.perf_counter(); s = ag.act(x, P0, Wv); tms.append(time.perf_counter() - t1); tv.append(ag.tvi); nits.append(ag.nit); tfl.append(ag.tfl)
        if os.environ.get('DBG') and int(t / TAU) % 5 == 0: print('  t %.1f x %s a %d W0x/Wx %.2f/%.2f Wq %s cl %s ok %s' % (t, np.round(x, 2), s, ag.W0x, ag.Wx, np.round(getattr(ag, 'Wq', np.zeros(1)), 2).tolist(), np.round(getattr(ag, 'cl', np.zeros(1)), 2).tolist(), getattr(ag, 'okc', np.zeros(1)).astype(int).tolist()), flush=True)
        PT = H.pos(t + (np.arange(nsub) + 1) * .002); xn, yn, thn, vn, m, hit = execute_dyn(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], .002, nsub, PT); mn = min(mn, m); x = np.array([xn, yn, thn, vn]); traj.append(x.copy())
        if hit >= 0: return t + hit, mn, float(np.mean(tms)), float(np.mean(tv)), float(np.mean(nits)), float(np.mean(tfl))
        t += TAU
    return None, mn, float(np.mean(tms)), float(np.mean(tv)), float(np.mean(nits)), float(np.mean(tfl))

def tail_main():
    """диагностика хвоста: TAIL='k:i,k:i,…' — перестраиваем дерево k, гоним эпизоды, печатаем тип/d_near/V(x0)/T_ref"""
    import gc
    COMP = json.load(open(os.environ['COMPACT'])); A = np.load(os.path.join(HERE, 'starts100.npy'))[:, :4]
    want = {}
    for p in os.environ['TAIL'].split(','): k_, i_ = p.split(':'); want.setdefault(int(k_), []).append(int(i_))
    recs = []
    for k in sorted(want):
        S = mkS(k); back = build_back(S, TAU, NB, RB); ag = AgentDyn(S, back); H = D.Hazards('random', k); ref = {r['i']: r for r in COMP[str(k)]['ref']}
        dnear = lambda X: ag.tree.query(S.wrap(np.atleast_2d(X)) / S.scale, k=1)[0]
        ds = dnear(A[:100]); print('k=%d: d_near стартов (масштаб. ед.) мед %.3f p90 %.3f max %.3f' % (k, np.median(ds), np.percentile(ds, 90), ds.max()), flush=True)
        for i in want[k]:
            x0 = A[i]; ag.streak = 0; nt0 = getattr(ag, "ntie", 0); T, mn, *_ = episode(ag, x0, H); tr = np.array(ag.last_traj); dt = dnear(tr); v = np.abs(tr[:, 3]); rest = float((v < .05).mean())
            xy = tr[:, :2]; pl = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()); sd = float(np.linalg.norm(xy[0] - GOAL))
            dc = np.array([(np.linalg.norm(H.pos(j * TAU) - xy[j], axis=1) - HR).min() for j in range(len(xy))])
            typ = 'покой' if (T is None and rest > .5) or rest > .3 else ('таймаут' if T is None else ('объезд (путь/прямая %.1f, зазор до дисков мин %.2f)' % (pl / sd, dc.min()) if dc.min() < .45 else 'медленно (путь/прямая %.1f)' % (pl / sd)))
            W0x = float(np.min(ag.W0[ag.tree.query(S.wrap(x0[None]) / S.scale, k=min(KN * 3, ag.nP))[1] % ag.nP] + KV * ag.tree.query(S.wrap(x0[None]) / S.scale, k=min(KN * 3, ag.nP))[0]))
            rec = dict(k=k, i=i, T=T, Tref=ref[i]['T'], ratio=None if T is None else T / ref[i]['T'], W0x0=W0x, W0_over_Tref=W0x / ref[i]['T'], d_start=float(dnear(x0)[0]), d_max_traj=float(dt.max()), d_med_traj=float(np.median(dt)), d_final=float(dt[-1]), rest_frac=rest, ties=getattr(ag, 'ntie', 0) - nt0, steps=len(tr), type=typ, final=np.round(tr[-1], 2).tolist())
            recs.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
        ag = None; gc.collect()
    json.dump(recs, open('car_tail_%s.json' % os.environ.get('TAG', 'x'), 'w'), ensure_ascii=False)

if __name__ == '__main__':
    if MODE == 'tail': tail_main(); sys.exit(0)
    import gc
    S0 = np.load(os.path.join(HERE, 'b24', 'sel_starts.npy') if MODE == 'sel' else os.path.join(HERE, 'starts100.npy'))[:, :4]; NST = int(os.environ.get('NSEL', 32)) if MODE == 'sel' else 100; out = {}; ag = None
    COMP = json.load(open(os.environ['COMPACT'])) if MODE == 'test' else None
    for k in range(K0, K1):
        if MODE == 'test' or ag is None:   # test: сид k = дерево k + диски k (деревья 0–19 для подбора не использовались)
            ag = None; gc.collect(); ts_ = k if MODE == 'test' else TS; S = mkS(ts_); t0 = time.perf_counter(); back = build_back(S, TAU, NB, RB); tb = time.perf_counter() - t0; print('ts %d: build %.1f с, узлов %d' % (ts_, tb, len(back[0])), flush=True); ag = AgentDyn(S, back)
        H = D.Hazards('random', k)
        if MODE == 'test': ref = {r['i']: r for r in COMP[str(k)]['ref']}
        else: ref = {r['i']: r for r in json.load(open(os.path.join(HERE, 'b24', 'dynref', 'dynref_s%d.json' % k)))}
        res = []; dclr = 0.
        for i in range(NST):
            if os.environ.get('ONLY') and i not in [int(v) for v in os.environ['ONLY'].split(',')]: continue
            x0 = S0[i]; c0 = float((np.linalg.norm(H.pos(0.) - x0[:2], axis=1) - HR).min())
            if MODE == 'test': dclr = max(dclr, abs(c0 - ref[i]['clr0']))
            if c0 < 0: continue
            tref = ref[i]['T']
            if not tref: continue
            te = time.perf_counter(); T, mn, ms, tvi, nit_, tfl_ = episode(ag, x0, H); res.append((i, T, tref, mn, ms, tvi, time.perf_counter() - te, nit_, tfl_))
        ok = [r for r in res if r[1] is not None]; ratio = np.array([r[1] / r[2] for r in ok]); touch = sum(r[3] < 0 for r in res)
        print('DYN NB=%d KV=%g RB=%g TIE=%g DEP=%d [%s] диски k=%d: дошли %d/%d, касаний %d, T/T_ref мед %.3f p90 %.3f max %.3f, мс/шаг %.1f (VI %.1f, пометка %.1f, итераций %.0f), зазор мин %.3f, мс/эпизод %.0f, build %.1f с, макс.|clr0−ref| %.2e' % (
            NB, KV, RB, TIE, DEP, MODE, k, len(ok), len(res), touch, np.median(ratio), np.percentile(ratio, 90), ratio.max(), 1e3 * np.mean([r[4] for r in res]), 1e3 * np.mean([r[5] for r in res]), 1e3 * np.mean([r[8] for r in res]), np.mean([r[7] for r in res]), min(r[3] for r in res), 1e3 * np.mean([r[6] for r in res]), tb, dclr), flush=True)
        print('   не дошли:', [r[0] for r in res if r[1] is None], ' T/Tref>1.5:', [(r[0], round(r[1] / r[2], 2)) for r in ok if r[1] / r[2] > 1.5], ' тай-брейков (накоп.): %d' % getattr(ag, 'ntie', 0), flush=True); out[k] = res
        json.dump(out, open('car_dyn_%s_%d_%d_h%g_x%g_tie%g.json' % (MODE, K0, K1, HYST, HX, TIE), 'w'))
