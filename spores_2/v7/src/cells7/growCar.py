"""w24 (PLAN п.36): growCar = growN + SYS=car («статик кар»: x,y,θ,v; управление (a,ω); 3 диска; цель — круг по x,y). Копия growN (владелец B не затрагивается).
b2 (PLAN п.22): growN — grow3 для любой размерности n (SYS=dd n=3 | pend n=2 | manip n=4). grow3.py не трогать (линия A).
Клетка = ящик индексов в n−1 боковых осях × строки [ilo,ihi] на мелкой сетке KF^(n−1) сечения (гиперплоскость ⟂ f(p), оси e_k — правые сингулярные векторы P·J·P) × время;
рост в 2(n−1)+2 стороны по правилам grow3: изгиб (отклонение от полилинейной заплатки по 2^(n−1) углам, метрика W⁻¹) в эллипсоиде a²·T·W ≤ DELTA, наложение на соседа OVH·h.
Узлы M^(n−1) на строку (с гало), ячейки — гиперкубы 2^n вершин, индекс QB^n, обратная полилинейная карта Ньютоном n×n. По одному атласу на управление; агент каждые DTN берёт argmin."""
import numpy as np, os, sys, json, time, itertools
from tqdm import tqdm
import finish_gen as FG
E = os.environ.get
SYS = E('SYS', 'dd'); M = int(E('M', 5)); BIG = 1e3; HALO = .1; EPSJ = 1e-5; PI2 = 2 * np.pi
DTN = float(E('DTN', .1)); RMAX = float(E('RMAX', .3)); TMAX = float(E('TMAX', 3.)); DELTA = float(E('DELTA', .03)); KF = int(E('KF', 21 if SYS == 'dd' else 11 if SYS in ('manip', 'di4', 'car') else 41))
OVH = float(E('OVH', 2.)); FRAC = float(E('FRAC', .95)); RMIN = float(E('RMIN', .02)); MINROWS = int(E('MINROWS', 1)); GNEAR = float(E('GNEAR', .7)); NFAIL = int(E('NFAIL', 400)); QB = float(E('QB', .25))
BEPS = float(E('BEPS', .01)); GOALB = int(E('GOALB', 1)); GLIM = float(E('GLIM', .25)); TQ = float(E('TQDM_MI', 10)); MAXC = int(E('MAXC', 10 ** 9)); GS = int(E('GS', 300 if SYS in ('manip', 'car') else 0)); PESS = float(E('PESS', -1)); WTHR = float(E('WTHR', .5)); VF = float(E('VF', 0.)); FTMAX = float(E('FTMAX', 2.5))   # VF > 0: финиш стрельбой ≤3 дуг (finish_gen, research-17) один раз на старт при V* ≤ VF
# ---- системы: N, PERIOD (0 = нет), RHOV (полуширины цели), XLV (границы поля по непериодическим), US, f(y,u), Bq(y) = B·Bᵀ при |δu| ≤ 1 по каналам ----
if SYS == 'dd':
    N = 3; PERIOD = np.array([0, 0, PI2]); RHO = float(E('RHO', .05)); RHOV = np.full(3, RHO); XL = float(E('XL', 2.5)); XLV = np.array([XL, XL, 0.])
    US = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))
    def f(y, u): th = y[..., 2]; return np.stack([u[0] * np.cos(th), u[0] * np.sin(th), u[1] + 0 * th], -1)
    def Bq(th): c, s = np.cos(th[2]), np.sin(th[2]); return np.array([[c * c, c * s, 0], [c * s, s * s, 0], [0, 0, 1.]])
elif SYS == 'pend':
    N = 2; PERIOD = np.array([PI2, 0]); UM = float(E('UM', .3)); RHO = float(E('RHO', .1)); RHOV = np.full(2, RHO); XLV = np.array([0., float(E('WS', 3.5))]); US = (-UM, 0., UM)
    def f(y, u): return np.stack([y[..., 1], np.sin(y[..., 0]) + u], -1)
    def Bq(y): return np.array([[0, 0], [0, UM * UM]])
elif SYS == 'manip':
    N = 4; PERIOD = np.array([PI2, PI2, 0, 0]); RHOV = np.array([.3, .3, .5, .5]); WM = float(E('WM', 3.)); XLV = np.array([0., 0., WM, WM]); US = ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
    AA, BB, DD = 2.5, .5, 1.
    def accel(x, tau):
        c, s = np.cos(x[..., 1]), np.sin(x[..., 1]); w1, w2 = x[..., 2], x[..., 3]; h = -BB * s
        r1 = tau[0] - h * (2 * w1 * w2 + w2 ** 2); r2 = tau[1] + h * w1 ** 2   # b4 (п.35): знак Кориолиса по Лагранжу (был обратный: энергия ×350 при τ=0)
        m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD * np.ones_like(c); det = m11 * m22 - m12 ** 2
        return np.stack([(m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det], -1)
    def f(y, u): return np.concatenate([y[..., 2:], accel(y, u)], -1)
    def Bq(y):
        c = np.cos(y[1]); m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD; det = m11 * m22 - m12 ** 2; Mi = np.array([[m22, -m12], [-m12, m11]]) / det; Bm = np.zeros((4, 2)); Bm[2:] = Mi; return Bm @ Bm.T
elif SYS == 'di4':                                                                                          # w22 (PLAN п.24): двойной интегратор по двум осям, ẍ₁ = u₁, ẍ₂ = u₂, |uᵢ| ≤ 1; состояние (x₁, x₂, v₁, v₂)
    N = 4; PERIOD = np.zeros(4); RHO = float(E('RHO', .05)); RHOV = np.full(4, RHO); XL = float(E('XL', 2.5)); XLV = np.full(4, XL); US = ((1., 1.), (1., -1.), (-1., 1.), (-1., -1.))
    def f(y, u): return np.concatenate([y[..., 2:], np.broadcast_to(np.asarray(u, float), y[..., :2].shape)], -1)
    def Bq(y): return np.diag([0., 0., 1., 1.])
elif SYS == 'car':                                                                                          # w24 (п.36, static_car.md): ẋ=v cosθ, ẏ=v sinθ, θ̇=ω, v̇=a−k v; |a|≤1.5, |ω|≤1.8; v∈[−1,2], |x|,|y|≤5; 3 диска r .7; цель — круг (4,3.6) r .45
    N = 4; PERIOD = np.array([0, 0, PI2, 0]); KV = .625; UA, UW = 1.5, 1.8; VLO, VHI = -1., 2.; XL = float(E('XL', 5.)); XLV = np.array([XL, XL, 0., 0.]); GC = np.array([4., 3.6]); GR = .45
    RHOV = np.array([GR, GR, np.pi, 1.5]); DISCS = [(0., .2, .7), (1.8, 1.5, .7), (-1.6, 2.3, .7)]; US = tuple((a, w) for a in (UA, -UA) for w in (UW, -UW)) + (((UA, 0.), (-UA, 0.)) if int(E('NL', 6)) == 6 else ())
    def f(y, u): return np.stack([y[..., 3] * np.cos(y[..., 2]), y[..., 3] * np.sin(y[..., 2]), u[1] + 0 * y[..., 2], u[0] - KV * y[..., 3]], -1)
    def Bq(y): return np.diag([0., 0., UW * UW, UA * UA])
else: raise SystemExit('SYS?')
if SYS != 'car': DISCS = []
def dobs(y): return np.min([np.hypot(y[..., 0] - cx, y[..., 1] - cy) - r for cx, cy, r in DISCS], 0) if DISCS else np.full(np.shape(y[..., 0]), 9.)
m_ = N - 1; PER = PERIOD > 0; NP_ = np.flatnonzero(PER)
SH = np.array([[0. if k not in NP_ else s_[list(NP_).index(k)] for k in range(N)] for s_ in itertools.product(*[(0., PERIOD[k], -PERIOD[k]) for k in NP_])]) if len(NP_) else np.zeros((1, N)); KSH = len(SH)
def wrapy(y):
    y = np.array(y, float)
    for k in NP_: y[..., k] = (y[..., k] + PERIOD[k] / 2) % PERIOD[k] - PERIOD[k] / 2
    return y
def rk4(y, u, h, n):
    for _ in range(n):
        k1 = f(y, u); k2 = f(y + h / 2 * k1, u); k3 = f(y + h / 2 * k2, u); k4 = f(y + h * k3, u); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
def step(y, u, sg=1.): return rk4(y, u, sg * DTN / 2, 2)
def ingoal(y):
    if SYS == 'car': return np.hypot(y[..., 0] - GC[0], y[..., 1] - GC[1]) <= GR + 1e-9
    return (np.abs(wrapy(y)) <= RHOV + 1e-9).all(-1)
_FL = ~PER
def inbox(y):
    if SYS == 'car': return (np.abs(y[..., 0]) <= XLV[0]) & (np.abs(y[..., 1]) <= XLV[0]) & (y[..., 3] >= VLO) & (y[..., 3] <= VHI) & (dobs(y) > 0)
    return (np.abs(y[..., _FL]) <= XLV[_FL]).all(-1)
GM = float(E('GM', 0.)); SIDE = int(E('SIDE', 1)); COVTOL = float(E('COVTOL', 0.)); COVN = int(E('COVN', 20)); COVP = int(E('COVP', 200))
def inbox_g(y):
    if SYS == 'car': return (np.abs(y[..., 0]) <= XLV[0] + GM) & (np.abs(y[..., 1]) <= XLV[0] + GM) & (y[..., 3] >= VLO - GM) & (y[..., 3] <= VHI + GM) & (dobs(y) > 0)
    return (np.abs(y[..., _FL]) <= XLV[_FL] + GM).all(-1)                                       # w22 (r17 GM): клетки растут за край поля на GM, посев — только внутри
def jac(x, u): return np.stack([(f(x + EPSJ * e, u) - f(x - EPSJ * e, u)) / (2 * EPSJ) for e in np.eye(N)], 1)
def wstep(W, A, Bm, h, sg):
    A = sg * A; Ph = np.eye(N) + h * A + h * h / 2 * A @ A
    return Ph @ W @ Ph.T + h * Bm + h * h / 2 * (A @ Bm + Bm @ A.T) + h ** 3 / 3 * A @ Bm @ A.T
def basis(p, u):
    """оси сечения (m×n): правые сингулярные векторы P·J·P (убывание), J = 0 — проекции осей координат; Грам–Шмидт в f⊥."""
    f0 = f(p, u); fh = f0 / np.linalg.norm(f0); P = np.eye(N) - np.outer(fh, fh); S, Vt = np.linalg.svd(P @ jac(p, u) @ P)[1:]
    cand = (list(Vt[:m_]) if S[0] >= 1e-9 else []) + list(np.eye(N)); out = []
    for v in cand:
        v = P @ v
        for e in out: v = v - (v @ e) * e
        nv = np.linalg.norm(v)
        if nv > 1e-6: out.append(v / nv)
        if len(out) == m_: break
    return np.array(out)
def shoot_pol(y, f, US, RHOV, wrapy, tmax, NA=3, inits=2):
    """research-22 fin2.py: сетка shoot_grid (FH .1) + доводка SLSQP только лучшей топологии (2 старта)."""
    from scipy.optimize import minimize
    T0, tp = shoot_grid(y, f, US, RHOV, wrapy, tmax, NA)
    if tp is None: return T0, tp
    R = .9 * np.asarray(RHOV); best = (T0, tp); n = len(tp)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FG.flow(z, US[k], max(dt, 0.), f)
        return wrapy(z)
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if r.success and np.all(np.abs(end(r.x)) <= np.asarray(RHOV)) and r.x.sum() < best[0]: best = (float(r.x.sum()), tp)
    return best
def _wcar(z): z = np.asarray(z); return np.stack([z[..., 0] - GC[0], z[..., 1] - GC[1], 0 * z[..., 0], 0 * z[..., 0]], -1) if SYS == 'car' else wrapy(z)   # w24: финиш стрельбой для цели-круга (вписанный квадрат, θ и v свободны); диски у цели далеко (>3)
FH = float(E('FH', .1))   # w24 (research-22 п.42): быстрый финиш — перебор времён ≤ NA дуг на сетке FH пачкой rk4 вместо SLSQP (в десятки раз быстрее)
def shoot_grid(y, f, US, RHOV, wrapy, tmax, NA=3, inits=2):
    t0 = time.time(); K = int(round(tmax / FH)); R = .9 * np.asarray(RHOV); best = (np.inf, None); nu = len(US)
    def traj(Y, u):
        out = [Y]
        for _ in range(K): Y = rk4(Y, u, FH, 1); out.append(Y)
        return np.stack(out, 1)                                                                # (B, K + 1, N), время j·H
    def upd(S, T, tp):
        nonlocal best
        ok = (np.abs(wrapy(S)) <= R).all(-1)
        if ok.any():
            t = np.where(ok, T, np.inf).min()
            if t < best[0]: best = (float(t), tp)
    tj = np.arange(K + 1) * FH; L1 = {a: traj(np.asarray(y, float)[None], US[a]) for a in range(nu)}
    for a in range(nu): upd(L1[a], tj[None], (a,))
    if NA >= 2:
        for a in range(nu):
            S1 = L1[a][0, 1:]; T1 = tj[1:]
            for b in range(nu):
                if b == a: continue
                S2 = traj(S1, US[b]); T2 = T1[:, None] + tj[None]; upd(S2, T2, (a, b))
                if NA >= 3:
                    m = T2[:, 1:] < min(best[0], tmax); S2f = S2[:, 1:][m]; T2f = T2[:, 1:][m]
                    for c in range(nu):
                        if c == b or not len(S2f): continue
                        S3 = traj(S2f, US[c]); upd(S3, T2f[:, None] + tj[None], (a, b, c))
    return best
def goal_dist(P):
    """расстояние до поверхности коробки цели (>0 снаружи и внутри — модуль), по Евклиду."""
    if SYS == 'car': return np.abs(np.hypot(P[..., 0] - GC[0], P[..., 1] - GC[1]) - GR)
    d = np.abs(wrapy(P)) - RHOV; out = np.linalg.norm(np.maximum(d, 0), axis=-1); ins = -d.max(-1); return np.where(d.max(-1) > 0, out, ins)
def nearb(P):
    if SYS == 'car': return (goal_dist(P) < BEPS) | (np.abs(dobs(P)) < BEPS) if GOALB else np.abs(dobs(P)) < BEPS   # w24: поверхности дисков — стенки роста
    return np.zeros(len(P), bool) if not GOALB else goal_dist(P) < BEPS
LAT = (slice(None),) * m_
class Cell:
    def __init__(s, c, u, r, e): s.c, s.u, s.r, s.e = np.array(c, float), u, np.array(r, float), e; s.nb = 0; s.nf = 0
    def build(s):
        """сетка узлов G (nt, M×m, n): заплатка с гало, пронесённая потоком на nb шагов назад и nf вперёд."""
        ax = [np.linspace(-(1 + HALO) * r, (1 + HALO) * r, M) for r in s.r]; seg = np.broadcast_to(s.c, (M,) * m_ + (N,)).copy()
        for k in range(m_): sh = [1] * m_ + [1]; sh[k] = M; seg = seg + ax[k].reshape(sh) * s.e[k]
        fw = [seg]; bw = []; y = seg
        for _ in range(s.nf): fw.append(step(fw[-1], s.u))
        for _ in range(s.nb): y = step(y, s.u, -1.); bw.append(y)
        s.G = np.array(bw[::-1] + fw)
LET = 'abcdefgh'[:N]; VOFF = np.array(list(itertools.product((0, 1), repeat=N)))                         # вершины гиперячейки: (строка, боковые…)
EIN = 'k' + LET + 'z,' + ','.join('k' + c for c in LET) + '->kz'
def _contract(Xa, sa, j=-1):
    """23а (research-18): значение (K, N) полилинейной карты гиперячейки в sa (K, N) свёрткой по осям; j ≥ 0 — производная по оси j. Вершины — порядок VOFF."""
    T = Xa.reshape((len(Xa),) + (2,) * N + (N,))
    for k in range(N): T = (T[:, 1] - T[:, 0]) if k == j else T[:, 0] + sa[:, k].reshape((-1,) + (1,) * (N - k)) * (T[:, 1] - T[:, 0])
    return T
class HexIdx:
    """индекс гиперячеек: бины QB^n (периодические оси — копии ±период), запрос — все точки сразу; полилинейная обратная карта Ньютоном n×n."""
    def __init__(s): s.n = 0; s.cap = 0; s.X = np.zeros((0, 2 ** N, N)); s.LO = np.zeros((0, N)); s.HI = np.zeros((0, N)); s.I = np.zeros((0, N + 1), np.int32); s.bins = {}; s.R = []; s.nc = 0
    def add(s, c):
        g = c.G; nt = g.shape[0]; ns = (nt - 1,) + (M - 1,) * m_
        X = np.stack([g[tuple(slice(d, n_ + d) for d, n_ in zip(o, ns))] for o in VOFF], -2).reshape(-1, 2 ** N, N); n = len(X)
        ii = [a.ravel() for a in np.indices(ns)]
        if s.n + n > s.cap:
            cap = max(2 * s.cap, s.n + n, 1 << 16); X2 = np.zeros((cap, 2 ** N, N)); I2 = np.zeros((cap, N + 1), np.int32); X2[:s.n] = s.X[:s.n]; I2[:s.n] = s.I[:s.n]; L2 = np.zeros((cap, N)); H2 = np.zeros((cap, N)); L2[:s.n] = s.LO[:s.n]; H2[:s.n] = s.HI[:s.n]; s.X, s.I, s.LO, s.HI, s.cap = X2, I2, L2, H2, cap
        s.X[s.n:s.n + n] = X; s.I[s.n:s.n + n] = np.c_[(np.full(n, s.nc),) + tuple(ii)]; hid = np.arange(s.n, s.n + n); s.n += n; s.R.append(c.r); s.nc += 1
        lo = X.min(1) - 1e-9; hi = X.max(1) + 1e-9; s.LO[s.n - n:s.n] = lo; s.HI[s.n - n:s.n] = hi; Ks, Vs = [], []
        for si, sh in enumerate(SH):
            l, h = lo + sh, hi + sh; v = np.ones(n, bool)
            for k in NP_: v &= (h[:, k] >= -PERIOD[k] / 2 - QB) & (l[:, k] <= PERIOD[k] / 2 + QB)
            if not v.any(): continue
            i0 = np.floor(l[v] / QB).astype(np.int64); i1 = np.floor(h[v] / QB).astype(np.int64); hv = hid[v]; sp = (i1 - i0).max(0)
            for dd_ in itertools.product(*[range(a + 1) for a in sp]):
                dd_ = np.array(dd_); k = ((i0 + dd_) <= i1).all(1)
                if k.any(): Ks.append(s._key(i0[k] + dd_)); Vs.append(hv[k] * KSH + si)
        if not Ks: return
        K = np.concatenate(Ks); V = np.concatenate(Vs); o = np.argsort(K, kind='stable'); K, V = K[o], V[o]; uk, st = np.unique(K, return_index=True); st = np.r_[st, len(K)]
        for k_, a_, b_ in zip(uk.tolist(), st[:-1], st[1:]): s.bins.setdefault(k_, []).append(V[a_:b_])
    @staticmethod
    def _key(ib):
        key = np.zeros(len(ib), np.int64)
        for k in range(N): key = key * 1024 + (ib[:, k] + 512)
        return key
    def query(s, Y):
        """пары (точка, гиперячейка): pi, hid, sc (n координат в ячейке: время, боковые)."""
        z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, N)))
        if s.n == 0 or not len(Y): return z
        kk = s._key(np.floor(Y / QB).astype(np.int64)); o = np.argsort(kk, kind='stable'); uk, st = np.unique(kk[o], return_index=True); st = np.r_[st, len(o)]
        PI, VV, out, cnt = [], [], [], 0; lim = int(1.5e6 * 24 / (2 ** N * N))
        for k_, a_, b_ in zip(uk.tolist(), st[:-1], st[1:]):
            lst = s.bins.get(k_)
            if not lst: continue
            if len(lst) > 1: lst = [np.concatenate(lst)]; s.bins[k_] = lst
            arr = lst[0]; pts = o[a_:b_]; PI.append(np.repeat(pts, len(arr))); VV.append(np.tile(arr, len(pts))); cnt += len(pts) * len(arr)
            if cnt > lim: out.append(s._test(Y, np.concatenate(PI), np.concatenate(VV))); PI, VV, cnt = [], [], 0
        if PI: out.append(s._test(Y, np.concatenate(PI), np.concatenate(VV)))
        if not out: return z
        return [np.concatenate(a) for a in zip(*out)]
    def _test_orig(s, Y, pi, vv):
        hid = vv // KSH; y = Y[pi] - SH[vv % KSH]; ok = ((y >= s.LO[hid]) & (y <= s.HI[hid])).all(1); pi, hid, y = pi[ok], hid[ok], y[ok]
        z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, N)))
        if not len(pi): return z
        X = s.X[hid]; sc = np.full((len(pi), N), .5); alive = np.arange(len(pi)); dw = np.array([-1., 1.])
        def wm(w, j=-1):
            K = len(w[0]); dk = np.broadcast_to(dw, (K, 2)); out = dk if j == 0 else w[0]
            for k in range(1, N): out = (out[:, :, None] * (dk if j == k else w[k])[:, None, :]).reshape(K, -1)
            return out
        for it in range(8):
            Xa, ya, sa = X[alive], y[alive], sc[alive]; w = [np.stack([1 - sa[:, k], sa[:, k]], 1) for k in range(N)]
            F = np.einsum('kv,kvz->kz', wm(w), Xa) - ya; J = np.stack([np.einsum('kv,kvz->kz', wm(w, j), Xa) for j in range(N)], 2)
            try: d = np.linalg.solve(J, F[..., None])[..., 0]
            except np.linalg.LinAlgError: d = np.linalg.solve(J + 1e-12 * np.eye(N), F[..., None])[..., 0]
            sc[alive] = sa - d
            if it in (0, 2):                                                             # точка внутри почти аффинной ячейки сходится сразу; вылетевшие за запас — отброшены
                lim = 1.0 if it == 0 else .2; alive = alive[((sc[alive] >= -lim) & (sc[alive] <= 1 + lim)).all(1)]
                if not len(alive): return z
        sc = sc[alive]; hid, X, y, pi = hid[alive], X[alive], y[alive], pi[alive]; w = [np.stack([1 - sc[:, k], sc[:, k]], 1) for k in range(N)]; F = np.einsum('kv,kvz->kz', wm(w), X) - y; tol = 1e-7
        k = (np.linalg.norm(F, axis=1) < 1e-7) & ((sc >= -tol) & (sc <= 1 + tol)).all(1)
        return pi[k], hid[k], np.clip(sc[k], 0, 1)
    def _prep(s):
        """23а (research-18, r18/affine_pre.py): центр A0 и J⁻¹ в центре на гиперячейку, инкрементально по s.n."""
        p = getattr(s, '_pn', 0)
        if p == s.n: return
        X = s.X[p:s.n]; h = np.full((len(X), N), .5); A0 = _contract(X, h); J = np.stack([_contract(X, h, j) for j in range(N)], 2); JI = np.linalg.pinv(J)
        s.A0 = A0 if p == 0 else np.concatenate([s.A0[:p], A0]); s.JI = JI if p == 0 else np.concatenate([s.JI[:p], JI]); s._pn = s.n
    def _test(s, Y, pi, vv, tolc=1e-10):
        """FASTT=0 — старый путь (_test_orig); иначе аффинный предфильтр sc0 ∈ [−MG, 1+MG] + Ньютон свёрткой по осям (множество пар то же, ×2 у research-18)."""
        if not int(E('FASTT', 1)): return s._test_orig(Y, pi, vv)
        s._prep(); hid = vv // KSH; y = Y[pi] - SH[vv % KSH]; ok = ((y >= s.LO[hid]) & (y <= s.HI[hid])).all(1); pi, hid, y = pi[ok], hid[ok], y[ok]
        sc0 = .5 + np.einsum('kij,kj->ki', s.JI[hid], y - s.A0[hid]); MG = float(E('MG', .4)); ok = ((sc0 >= -MG) & (sc0 <= 1 + MG)).all(1); pi, hid, y, sc = pi[ok], hid[ok], y[ok], sc0[ok]
        z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, N)))
        if not len(pi): return z
        X = s.X[hid]; alive = np.arange(len(pi))
        for it in range(8):
            Xa, ya, sa = X[alive], y[alive], sc[alive]; F = _contract(Xa, sa) - ya; J = np.stack([_contract(Xa, sa, j) for j in range(N)], 2)
            try: d = np.linalg.solve(J, F[..., None])[..., 0]
            except np.linalg.LinAlgError: d = np.linalg.solve(J + 1e-12 * np.eye(N), F[..., None])[..., 0]
            sc[alive] = sa - d; keep = ((sc[alive] >= -.2) & (sc[alive] <= 1.2)).all(1) if it >= 1 else np.ones(len(alive), bool)
            bad = alive[~keep]; sc[bad] = 9.; alive = alive[keep & (np.abs(d).max(1) >= tolc)]
            if not len(alive): break
        F = _contract(X, sc.clip(-1, 2)) - y; tol = 1e-7
        k = (np.linalg.norm(F, axis=1) < 1e-7) & ((sc >= -tol) & (sc <= 1 + tol)).all(1)
        return pi[k], hid[k], np.clip(sc[k], 0, 1)
    def covered(s, Y):
        Y = wrapy(np.atleast_2d(Y)); m = np.zeros(len(Y), bool); pi, hid, sc = s.query(Y)
        if len(pi):
            R = np.array(s.R); cid = s.I[hid, 0]; J = s.I[hid, 2:]; w = 2 * (1 + HALO) / (M - 1)
            sl = -(1 + HALO) * R[cid] + (J + sc[:, 1:]) * w * R[cid]; core = (np.abs(sl) <= R[cid] + 1e-9).all(1); m[pi[core]] = True
        return m
def mlin(Pb, n_):
    """полилинейная заплатка по 2^m углам ящика Pb (строки, n_1..n_m, n)."""
    C = Pb
    for j in range(m_): C = np.take(C, [0, -1], axis=1 + j)
    D = C
    for j in range(m_):
        sh = [1] * (m_ + 2); sh[1 + j] = n_[j]; a = np.linspace(0, 1, n_[j]).reshape(sh); D = (1 - a) * np.take(D, [0], axis=1 + j) + a * np.take(D, [1], axis=1 + j)
    return D
import collections; STOP = collections.Counter()                                                          # w22: причины остановки роста (STOPS=1 печатает по слою)
def growN(p, u, idx, rm, tm):
    """клетка = ящик индексов; направление (ось k ±, F, B) растёт, пока: в поле, изгиб в эллипсоиде ≤ DELTA, не барьер; упёрлось в соседа (грань покрыта > FRAC) — добираем OVH·h и стоп."""
    e = basis(p, u); S = np.linspace(-rm, rm, KF); k0 = KF // 2; nmax = int(tm / DTN + 1e-9); lat = np.stack(np.meshgrid(*([S] * m_), indexing='ij'), -1); base = p + lat @ e; rows = {0: base}; Wt = {0: np.zeros((N, N))}
    for sg in (1, -1):
        y = base; yc = p.copy(); W = np.zeros((N, N))
        for i in range(1, nmax + 1):
            y = step(y, u, float(sg)); yc = step(yc, u, float(sg)); W = wstep(W, jac(yc, u), Bq(yc), DTN, sg); rows[sg * i] = y; Wt[sg * i] = W * (i * DTN)
            if not inbox_g(yc): break
    imin, imax = min(rows), max(rows); R = np.stack([rows[i] for i in range(imin, imax + 1)])
    def bend(klo, khi, ilo, ihi):
        n_ = [b - a + 1 for a, b in zip(klo, khi)]
        if max(n_) < 3: return 0.
        Wc = Wt[ihi] if ihi >= -ilo else Wt[ilo]; T_ = (ihi - ilo) * DTN; Wc = Wc * (T_ / max(max(ihi, -ilo) * DTN, 1e-9)); lam, Q = np.linalg.eigh(Wc); Wi = (Q / np.maximum(lam, 1e-12 * max(lam.max(), 1e-30))) @ Q.T
        Pb = R[(slice(ilo - imin, ihi - imin + 1),) + tuple(slice(a, b + 1) for a, b in zip(klo, khi))]; d = Pb - mlin(Pb, n_)
        return float(np.sqrt(np.max(np.einsum('...k,kl,...l->...', d, Wi, d))))
    klo = [k0 - 1] * m_; khi = [k0 + 1] * m_; ilo, ihi = 0, 0; dirs = [('a', k, sg) for k in range(m_) for sg in (1, -1)] + [('F',), ('B',)]; act = {d: True for d in dirs}; extra = {}
    def faceof(d, nk):
        sl = (slice(ilo - imin, ihi - imin + 1),) + tuple(slice(a, b + 1) for a, b in zip(klo, khi)); sl = list(sl); sl[1 + d[1]] = nk; return R[tuple(sl)].reshape(-1, N)
    while any(act.values()):
        for d in dirs:
            if not act[d]: continue
            if d[0] == 'a':
                if ihi - ilo < 3 and (act[('F',)] or act[('B',)]): continue
                k, pl = d[1], d[2] > 0; nk = khi[k] + 1 if pl else klo[k] - 1
                if nk < 0 or nk >= KF: act[d] = False; STOP['kf'] += 1; continue
                face = faceof(d, nk); nlo = list(klo); nhi = list(khi); nlo[k] = min(klo[k], nk); nhi[k] = max(khi[k], nk)
                fb = not inbox_g(face).all(); bb = (not fb) and bend(nlo, nhi, ilo, ihi) > DELTA; gb = (not fb and not bb) and GOALB and nearb(face).any()
                if fb or bb or gb: act[d] = False; STOP['field' if fb else 'bend' if bb else 'goal'] += 1; continue
                klo, khi = nlo, nhi
                if d in extra:
                    extra[d] -= 1
                    if extra[d] <= 0: act[d] = False; STOP['ovh'] += 1
                elif (idx.covered(face) | ~inbox_g(face)).mean() > FRAC: extra[d] = int(np.ceil(OVH * (khi[k] - klo[k]) / (M - 1)))
            else:
                i = ihi + 1 if d[0] == 'F' else ilo - 1
                if i < imin or i > imax: act[d] = False; STOP['rows'] += 1; continue
                face = R[(i - imin,) + tuple(slice(a, b + 1) for a, b in zip(klo, khi))].reshape(-1, N)
                fb = not inbox_g(face).all(); bb = (not fb) and bend(klo, khi, min(ilo, i), max(ihi, i)) > DELTA; gb = (not fb and not bb) and GOALB and nearb(face).any()
                if fb or bb or gb: act[d] = False; STOP['tfield' if fb else 'tbend' if bb else 'tgoal'] += 1; continue
                if d[0] == 'F': ihi = i
                else: ilo = i
                if d in extra: act[d] = False; STOP['tovh'] += 1
                elif (idx.covered(face) | ~inbox_g(face)).mean() > FRAC: extra[d] = 1
    r = np.array([(S[b] - S[a]) / 2 for a, b in zip(klo, khi)]); near = np.linalg.norm(wrapy(p)) < GNEAR
    if ihi - ilo < (1 if near else MINROWS) or r.min() < (.02 if near else RMIN): return None
    cen = p + sum(e[k] * (S[klo[k]] + S[khi[k]]) / 2 for k in range(m_)); c = Cell(cen, u, r / (1 + HALO), e); c.nf, c.nb = ihi, -ilo; return c
def glimits(p, u):
    d = float(np.hypot(p[0] - GC[0], p[1] - GC[1]) + .05) if SYS == 'car' else float(np.linalg.norm(wrapy(p))); return float(np.clip(GLIM * d, .04, RMAX)), float(np.clip(2 * d, .3, TMAX))
LIMITS = glimits if GLIM > 0 else None
def rand_seed(rng):
    if SYS == 'car': return np.array([rng.uniform(-XLV[0], XLV[0]), rng.uniform(-XLV[0], XLV[0]), rng.uniform(-np.pi, np.pi), rng.uniform(VLO, VHI)])
    return np.array([rng.uniform(-PERIOD[k] / 2, PERIOD[k] / 2) if PER[k] else rng.uniform(-XLV[k], XLV[k]) for k in range(N)])
def goal_seeds(rng):
    """GS точек на поверхности коробки цели (чуть снаружи): очередь FIFO растит клетки от цели назад по потоку (BFS), а не случайно по всему полю (4D: цель ≈ 3e-4 объёма)."""
    if SYS == 'car':
        out = []
        for _ in range(GS): a = rng.uniform(0, 2 * np.pi); out.append(np.array([GC[0] + 1.05 * GR * np.cos(a), GC[1] + 1.05 * GR * np.sin(a), rng.uniform(-np.pi, np.pi), rng.uniform(VLO, VHI)]))
        return out
    ar = np.array([np.prod(np.delete(RHOV, k)) for k in range(N)]); ar = ar / ar.sum(); out = []
    for _ in range(GS):
        k = rng.choice(N, p=ar); y = rng.uniform(-RHOV, RHOV); y[k] = rng.choice([-1., 1.]) * RHOV[k] * 1.05; out.append(y)
    return out
def build_layer(u, rng, log=None):
    cells = []; fails = 0; idx = HexIdx(); queue = goal_seeds(rng) if GS else []; bar = tqdm(total=NFAIL, desc='layer u=%s' % (u,), mininterval=TQ, leave=False); ctr = (M // 2,) * m_
    while fails < NFAIL and len(cells) < MAXC:
        if len(cells) % 10 == 0: bar.n = fails; bar.set_postfix(cells=len(cells), queue=len(queue)); bar.refresh()
        p = queue.pop(0) if queue else rand_seed(rng)
        if not inbox(p): continue
        p = wrapy(p)
        if ingoal(p): continue
        if idx.covered(p[None])[0]: fails += 0 if queue else 1; continue
        rm, tm = LIMITS(p, u) if LIMITS else (RMAX, TMAX); c = growN(p, u, idx, rm, tm)
        if c is None: fails += 0 if queue else 1; continue
        if E('STOPS') and len(cells) % 50 == 0: print('stops', len(cells), dict(STOP), flush=True)
        fails = 0; c.build(); cells.append(c); idx.add(c); g0, g1 = c.G[-1], c.G[0]; yf = g0[ctr]; yb = g1[ctr]
        for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): yf = step(yf, u); yb = step(yb, u, -1.)
        queue += [yf, yb]
        if SIDE:
            for e0 in (c.c, g0[ctr], g1[ctr]):
                for k in range(m_): queue += [e0 + 1.9 * c.r[k] * c.e[k], e0 - 1.9 * c.r[k] * c.e[k]]
        if COVTOL > 0 and len(cells) % COVN == 0:                                                               # w22 (r17): стоп по покрытию — доля непокрытых из COVP случайных проб < COVTOL
            pr = np.array([q for q in (rand_seed(rng) for _ in range(COVP)) if inbox(q) and not ingoal(q)])
            if E('COVDBG') and len(pr): print('cov', len(cells), 'непокрыто', round(1. - idx.covered(wrapy(pr)).mean(), 4), 'проб', len(pr), flush=True)
            if len(pr) and 1. - idx.covered(wrapy(pr)).mean() < COVTOL: print('layer', u, 'стоп по покрытию:', len(cells), 'клеток, непокрыто', round(1. - idx.covered(wrapy(pr)).mean(), 4), flush=True); break
        if log: log(u, cells)
    bar.close(); return cells, idx
def _layer(a): return build_layer(a[0], np.random.default_rng(a[1]), None)[0]
_PQ = None
def _dedup(pi, hid, sc):
    """DEDUP=K (b3): на точку оставить K ячеек с наибольшим запасом до границы (min по координатам min(sc, 1-sc)) — при FRAC 1 пар на узел много (память стенсилов)."""
    K = int(E('DEDUP', 0))
    if K <= 0 or not len(pi): return pi, hid, sc
    mg = np.minimum(sc, 1 - sc).min(1); o = np.lexsort((-mg, pi)); pi, hid, sc = pi[o], hid[o], sc[o]
    st = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]; rk = np.arange(len(pi)) - np.repeat(st, np.diff(np.r_[st, len(pi)]))
    k = rk < K; return pi[k], hid[k], sc[k]
def _pq_work(a): return tuple(_dedup(*_PQ.query(a[1])[:3])) + (a[0],)
def _pquery(qx, Y):
    """SPAR=k (b3): запрос индекса чанками в fork-пуле (стенсилы последовательны: 4 u × ~3M точек в одном процессе); без SPAR — как раньше."""
    global _PQ
    k = int(E('SPAR', 0))
    if k <= 1 or len(Y) < 50000: return _dedup(*qx.query(Y))
    from multiprocessing import Pool
    _PQ = qx; ch = np.array_split(np.arange(len(Y)), 4 * k)
    with Pool(k) as pool: R = pool.map(_pq_work, [(c[0], Y[c]) for c in ch if len(c)])
    return np.concatenate([r[0] + r[3] for r in R]), np.concatenate([r[1] for r in R]), np.concatenate([r[2] for r in R])
class Atlas:
    def __init__(s, seed=0, log=None):
        s.layers = []
        if int(E('PAR', 1)):
            from multiprocessing import Pool
            with Pool(len(US)) as pool: s.layers = pool.map(_layer, [(u, seed + k) for k, u in enumerate(US)])
        else: s.layers = [_layer((u, seed + k)) for k, u in enumerate(US)]
        s.finish()
    def finish(s):
        s.cells = [c for l in s.layers for c in l]; n_ = 0
        for c in s.cells: c.o = n_; n_ += c.G.shape[0] * M ** m_
        s.P = np.concatenate([c.G.reshape(-1, N) for c in s.cells]); s.N = n_; s.goal = ingoal(s.P); s.V = np.full(n_, BIG); s.V[s.goal] = 0.
    def stencils(s, Y):
        if not hasattr(s, 'qx'):
            s.qx = HexIdx()
            for c in s.cells: s.qx.add(c)
            s.O = np.array([c.o for c in s.cells])
        Y = wrapy(Y); pi, hid, sc = _pquery(s.qx, Y)
        if E('PAIRS'): print('PAIRS', len(Y), len(pi), round(len(pi) / max(len(Y), 1), 2), flush=True)
        if not len(pi): return np.zeros(0, int), np.zeros((0, 2 ** N), np.int64), np.zeros((0, 2 ** N), np.float32)
        stv = np.array([M ** (m_ - k) for k in range(N)]); off = VOFF @ stv; n_ = len(pi); idt = np.int32 if s.N < 2 ** 31 else np.int64
        IDX = np.empty((n_, 2 ** N), idt); W = np.empty((n_, 2 ** N), np.float32); CH = 1 << 21                       # чанками: 41M пар × 16 вершин × N в float64 не помещались в память (b3)
        for a in range(0, n_, CH):
            b = min(a + CH, n_); ii = s.qx.I[hid[a:b]].astype(np.int64); base = s.O[ii[:, 0]] + (ii[:, 1:] * stv).sum(1); IDX[a:b] = base[:, None] + off[None]
            sb = sc[a:b]; W[a:b] = np.prod(np.where(VOFF[None] == 1, sb[:, None, :], 1 - sb[:, None, :]), 2)
        return pi, IDX, W
    @staticmethod
    def interp(W, VI):
        if PESS < 0: v = (W * VI).sum(1); bad = ((W > 1e-6) & (VI >= BIG / 2)).any(1); v[bad] = BIG; return v
        ok = VI < BIG / 2; w = W * ok; sm = w.sum(1)                                                                  # 23б (research-18): PESS=Δ — неизвестная вершина = max известных + Δ; BIG только при весе известных < WTHR
        with np.errstate(invalid='ignore'): v = (w * np.where(ok, VI, 0.)).sum(1) + (1 - sm) * (np.where(ok, VI, -np.inf).max(1) + PESS)
        v[sm < WTHR] = BIG; return v
    def vstar(s, Y):
        I, IDX, W = s.stencils(Y); out = np.full(len(Y), BIG)
        if len(I): np.minimum.at(out, I, s.interp(W, s.V[IDX]))
        out[ingoal(Y)] = 0.; return out
    def tgoal(s, Y, u, n=8):
        tg = np.full(len(Y), np.inf); y = Y
        for i in range(1, n + 1): y = rk4(y, u, DTN / n, 1); tg = np.where(np.isinf(tg) & ingoal(y), DTN * i / n, tg)
        return tg
    def solve(s, it=20000):
        if E('LOADE'): I = np.load(E('LOADE') + '_I.npy'); IDX = np.load(E('LOADE') + '_IDX.npy'); W = np.load(E('LOADE') + '_W.npy'); Vg = np.load(E('LOADE') + '_Vg.npy')   # b4: кеш рёбер (стенсилы строятся ~10 мин) — для отладки solve
        else:
            I_, IDX_, W_ = [], [], []; Vg = np.full(s.N, np.inf)
            for u in US:
                Vg = np.minimum(Vg, s.tgoal(s.P, u)); a, b, c_ = s.stencils(step(s.P, u)); print('stencils built for u =', u, flush=True); I_.append(a); IDX_.append(b.astype(np.int32 if s.N < 2 ** 31 else np.int64)); W_.append(c_)
            I = np.concatenate(I_); del I_; IDX = np.concatenate(IDX_); del IDX_; W = np.concatenate(W_); del W_; o = np.argsort(I, kind='stable'); I = I[o]; IDX = IDX[o]; W = W[o]; del o   # b3: по одному, с освобождением (4u × 20M пар × 16 вершин не помещались)
            if E('SAVEE'): [np.save(E('SAVEE') + k_, v_) for k_, v_ in (('_I.npy', I), ('_IDX.npy', IDX), ('_W.npy', W), ('_Vg.npy', Vg))]
        st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; V = np.minimum(s.V, Vg)
        if int(E('SOLVEB', 0)):                                                                    # п.33 (research-21, прототип r18/solve_bucket.py; b4): вёдра V — узлы в порядке V, пересчёт только рёбер владельцев, чей вход изменился (коррекция меток, та же неподвижная точка)
            Nn = s.N; G_ = len(st); stE = np.r_[st, len(I)]; gpos = np.full(Nn, -1, np.int64); gpos[nd] = np.arange(G_); ch = 1 << 20; g_ = np.unique(np.r_[np.searchsorted(st, np.arange(0, len(I), ch)), G_]); keys = []; SBSELF = int(E('SBSELF', 1)); SBTOL = float(E('SBTOL', 1e-4)); Cc = np.full(len(I), DTN); nself = 0
            for g0, g1 in tqdm(list(zip(g_[:-1], g_[1:])), desc='обратная карта', mininterval=TQ, leave=False):
                a = st[g0]; b = st[g1] if g1 < G_ else len(I)
                if SBSELF:                                                                          # research-21 SBSELF: петля на себя (вершина стенсила = сам узел, вес s_) решается точно: V = (c + Σ_{j≠i} w_j V_j) / (1 − s_)
                    sm = (IDX[a:b] == I[a:b, None]) & (W[a:b] > 1e-6); s_ = (W[a:b] * sm).sum(1); kk = (s_ > 0) & (s_ < 1 - 1e-9); nself += int(kk.sum()); Wc = np.where(sm, 0., W[a:b]); Wc[kk] /= (1 - s_[kk])[:, None]; W[a:b] = Wc; cc = Cc[a:b]; cc[kk] = DTN / (1 - s_[kk]); cc[s_ >= 1 - 1e-9] = BIG
                w = W[a:b] > 1e-6; own = np.broadcast_to(I[a:b, None], w.shape)[w]; keys.append(np.unique(IDX[a:b][w].astype(np.int64) * Nn + own))   # вход → владелец (узел), по узлам: карта в ~20× компактнее, чем по рёбрам
            K_ = np.sort(np.concatenate(keys)); del keys; own_ = (K_ % Nn).astype(np.int32); ptr = np.searchsorted(K_ // Nn, np.arange(Nn + 1)); del K_
            D = float(E('BD', DTN)); fixed = s.goal; pend = V < BIG / 2; th = 0.; nb = 0; nev = 0; bar = tqdm(desc='solve вёдра', mininterval=TQ, leave=False)
            while pend.any():
                th = max(th, V[pend].min()); thr = th + D; nb += 1; bar.update(1); bt = np.flatnonzero(pend & (V < thr)); rd = 0; ne0 = nev; nb0 = len(bt)
                while len(bt):
                    rd += 1
                    pend[bt] = False; lo, hi = ptr[bt], ptr[bt + 1]; n_ = hi - lo; tot = int(n_.sum())
                    if not tot: break
                    ow = np.unique(own_[np.repeat(lo - np.r_[0, np.cumsum(n_)[:-1]], n_) + np.arange(tot)]); gi = gpos[ow]; ea_ = stE[gi]; cn = stE[gi + 1] - ea_; cs = np.r_[0, np.cumsum(cn)[:-1]]
                    e = np.repeat(ea_ - cs, cn) + np.arange(int(cn.sum())); nev += len(e); val = Cc[e] + s.interp(W[e], V[IDX[e]]); mv = np.minimum.reduceat(val, cs)
                    imp = (mv < V[ow] - SBTOL) & ~fixed[ow]; un = ow[imp]; V[un] = mv[imp]; pend[un] = True; bt = un[mv[imp] < thr]
                if nb <= 30 or nb % 20 == 0: print('ведро', nb, 'th %.2f' % th, 'узлов', nb0, 'раундов', rd, 'рёбер', nev - ne0, flush=True)
                th = thr
            print('solve вёдра: петель на себя', nself, 'SBTOL', SBTOL, 'вёдер', nb, 'вычислений рёбер', nev, 'на ребро %.2f' % (nev / len(I)), flush=True); s.V = V; s.n_it = nb; s.edges = len(I); return s
        bar = tqdm(total=it, desc='solve', mininterval=TQ, leave=False)
        for n in range(it):
            bar.update(1); val = np.empty(len(I))
            for a in range(0, len(I), 1 << 22): b = a + (1 << 22); val[a:b] = DTN + s.interp(W[a:b], V[IDX[a:b]])           # чанками: V[IDX] = пары × 16 float64 (десятки ГБ)
            new = V.copy(); new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st)); new[s.goal] = 0.
            d = np.max(np.abs(new - V)); V = new
            if d < 1e-9: break
        s.V = V; s.n_it = n; s.edges = len(I); return s
    def rollout(s, Q, tmax=40.):
        Y = wrapy(Q); n = len(Y); T = np.zeros(n); done = ingoal(Y); sw = np.zeros(n, int); pk = np.full(n, -1); path = [Y.copy()]; tried = np.zeros(n, bool); fin = np.zeros(n, bool)
        for _ in range(int(tmax / DTN)):
            if done.all(): break
            tgs = np.stack([s.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([s.vstar(step(Y, u)) for u in US], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
            if VF > 0:
                for i in np.flatnonzero(~done & ~tried & (J.min(1) <= VF)):
                    tried[i] = True; Ts, tp = ((shoot_pol if int(E('FPOL', 1)) else shoot_grid) if int(E('FGRID', 1)) and SYS == 'car' else FG.shoot)(Y[i], f, US, *((np.array([GR / np.sqrt(2)] * 2 + [9., 9.]), _wcar) if SYS == 'car' else (RHOV, wrapy)), FTMAX, NA=int(E('NA', 3)))
                    if np.isfinite(Ts): T[i] += Ts; done[i] = True; fin[i] = True; sw[i] += len(tp) - 1
            stuck = J.min(1) >= BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
            for ki, ui in enumerate(US):
                m = act & (k == ki)
                if not m.any(): continue
                hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
                for i in range(1, 9): y_i = rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
                Yn[m] = y8
            sw += act & (pk >= 0) & (k != pk); pk = np.where(act, k, pk); Y = wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= ingoal(Y) | stuck; path.append(Y.copy())
        T[~ingoal(Y) & ~fin] = np.inf; return T, sw, np.array(path)
HERE = os.path.dirname(os.path.abspath(__file__)); REF = os.path.join(HERE, '../../../v5chain/reports/research')
def starts_ref():
    """(Q, эталон T)."""
    if SYS == 'car':                                                                                        # research-21: 20 стартов спавна (r21/ref_car.starts), эталон = min по N 40/80
        rng = np.random.default_rng(0); S = []; HZ = np.array([c[:2] for c in DISCS])
        while len(S) < 20:
            p = rng.uniform([-4.25, -4.25], [0., .3])
            if np.linalg.norm(p - GC) < GR + .8 or (np.linalg.norm(HZ - p, axis=1) - .7).min() <= .2: continue
            d = GC - p; S.append(np.array([p[0], p[1], np.arctan2(d[1], d[0]) + rng.normal(0, .25), 0.]))
        return np.array(S), np.load(os.path.join(REF, 'r21/car_ref20.npy'))
    if SYS == 'dd': rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]; return Q, np.load(os.path.join(REF, 'dd_ref_60.npy'))
    if SYS == 'di4':                                                                                        # research-17: T* = max(T₁*, T₂*) в коробку ±.05 (r17/di4_ref.py)
        d_ = np.load(os.path.join(REF, 'r17/di4_ref_60.npy')); Q = np.random.default_rng(1).uniform(-2, 2, (60, 4)); assert np.allclose(Q, d_[:, :4]); return Q, d_[:, 6]
    if SYS == 'manip':
        rng = np.random.default_rng(0)
        for _ in range(32): rng.uniform(-1, 1, 4)
        Q = np.array([[*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)] for _ in range(8)]); ref = np.full(8, np.nan)
        for i in (0, 1, 3, 5): ref[i] = json.load(open(os.path.join(REF, 'manip_bruteforce_%d.json' % i)))[str(i)]['T']
        return Q, ref
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1); fb = os.path.join(REF, 'pend_ref_best_u%s.npy' % E('UM', '0.3'))
    return Q, np.load(fb if os.path.exists(fb) else os.path.join(HERE, 'pend_ref_T.npy'))
if __name__ == '__main__':
    t0 = time.time()
    if E('LOAD'): import pickle; d_ = pickle.load(open(E('LOAD'), 'rb')); A = Atlas.__new__(Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']; A.n_it = 0; A.edges = 0; tb = 0.; Q, ref = starts_ref()   # LOAD: атлас с V из DUMP — только rollout (напр. с VF)
    else:
        if E('LOADL'): import pickle; A = Atlas.__new__(Atlas); A.layers = pickle.load(open(E('LOADL'), 'rb'))['layers']; A.finish()   # w24: слои из DUMPL, solve заново
        else: A = Atlas()
        tb = time.time() - t0
        print('построено', [len(l) for l in A.layers], 'узлов', A.N, round(tb), 'с', flush=True)
        if E('DUMPL'): import pickle; pickle.dump(dict(layers=A.layers), open(E('DUMPL'), 'wb')); print('слои сохранены', E('DUMPL'), flush=True)   # w22: после потери 2-часового построения — слои на диск до solve
        t_s = time.time(); A.solve(); print('t_solve', round(time.time() - t_s, 1), 'it', A.n_it, flush=True)
        if E('DUMP'): import pickle; pickle.dump(dict(layers=A.layers, V=A.V), open(E('DUMP'), 'wb'))   # w24: дамп V ДО starts_ref (потеряли 20 мин solve из-за отсутствия файла эталона)
        Q, ref = starts_ref()
    T, sw, _ = A.rollout(Q); fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]
    print(json.dumps(dict(SYS=SYS, DTN=DTN, RMAX=RMAX, TMAX=TMAX, DELTA=DELTA, cells=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), big_nodes=round(float((A.V >= BIG / 2).mean()), 3), reach=round(float(np.isfinite(T).mean()), 3),
                          T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)) if fz.any() else None, sec_build=round(tb, 1), sec=round(time.time() - t0, 1))), flush=True)
