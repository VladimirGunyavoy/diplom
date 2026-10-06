"""b2 (PLAN п.22): growN — grow3 для любой размерности n (SYS=dd n=3 | pend n=2 | manip n=4). grow3.py не трогать (линия A).
Клетка = ящик индексов в n−1 боковых осях × строки [ilo,ihi] на мелкой сетке KF^(n−1) сечения (гиперплоскость ⟂ f(p), оси e_k — правые сингулярные векторы P·J·P) × время;
рост в 2(n−1)+2 стороны по правилам grow3: изгиб (отклонение от полилинейной заплатки по 2^(n−1) углам, метрика W⁻¹) в эллипсоиде a²·T·W ≤ DELTA, наложение на соседа OVH·h.
Узлы M^(n−1) на строку (с гало), ячейки — гиперкубы 2^n вершин, индекс QB^n, обратная полилинейная карта Ньютоном n×n. По одному атласу на управление; агент каждые DTN берёт argmin."""
import numpy as np, os, sys, json, time, itertools
from tqdm import tqdm
E = os.environ.get
SYS = E('SYS', 'dd'); M = int(E('M', 5)); BIG = 1e3; HALO = .1; EPSJ = 1e-5; PI2 = 2 * np.pi
DTN = float(E('DTN', .1)); RMAX = float(E('RMAX', .3)); TMAX = float(E('TMAX', 3.)); DELTA = float(E('DELTA', .03)); KF = int(E('KF', 21 if SYS == 'dd' else 11 if SYS == 'manip' else 41))
OVH = float(E('OVH', 2.)); FRAC = float(E('FRAC', .95)); RMIN = float(E('RMIN', .02)); MINROWS = int(E('MINROWS', 1)); GNEAR = float(E('GNEAR', .7)); NFAIL = int(E('NFAIL', 400)); QB = float(E('QB', .25))
BEPS = float(E('BEPS', .01)); GOALB = int(E('GOALB', 1)); GLIM = float(E('GLIM', .25)); TQ = float(E('TQDM_MI', 10)); MAXC = int(E('MAXC', 10 ** 9))
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
        r1 = tau[0] + h * (2 * w1 * w2 + w2 ** 2); r2 = tau[1] - h * w1 ** 2
        m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD * np.ones_like(c); det = m11 * m22 - m12 ** 2
        return np.stack([(m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det], -1)
    def f(y, u): return np.concatenate([y[..., 2:], accel(y, u)], -1)
    def Bq(y):
        c = np.cos(y[1]); m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD; det = m11 * m22 - m12 ** 2; Mi = np.array([[m22, -m12], [-m12, m11]]) / det; Bm = np.zeros((4, 2)); Bm[2:] = Mi; return Bm @ Bm.T
else: raise SystemExit('SYS?')
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
def ingoal(y): return (np.abs(wrapy(y)) <= RHOV + 1e-9).all(-1)
_FL = ~PER
def inbox(y): return (np.abs(y[..., _FL]) <= XLV[_FL]).all(-1)
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
def goal_dist(P):
    """расстояние до поверхности коробки цели (>0 снаружи и внутри — модуль), по Евклиду."""
    d = np.abs(wrapy(P)) - RHOV; out = np.linalg.norm(np.maximum(d, 0), axis=-1); ins = -d.max(-1); return np.where(d.max(-1) > 0, out, ins)
def nearb(P): return np.zeros(len(P), bool) if not GOALB else goal_dist(P) < BEPS
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
class HexIdx:
    """индекс гиперячеек: бины QB^n (периодические оси — копии ±период), запрос — все точки сразу; полилинейная обратная карта Ньютоном n×n."""
    def __init__(s): s.n = 0; s.cap = 0; s.X = np.zeros((0, 2 ** N, N)); s.I = np.zeros((0, N + 1), np.int32); s.bins = {}; s.R = []; s.nc = 0
    def add(s, c):
        g = c.G; nt = g.shape[0]; ns = (nt - 1,) + (M - 1,) * m_
        X = np.stack([g[tuple(slice(d, n_ + d) for d, n_ in zip(o, ns))] for o in VOFF], -2).reshape(-1, 2 ** N, N); n = len(X)
        ii = [a.ravel() for a in np.indices(ns)]
        if s.n + n > s.cap:
            cap = max(2 * s.cap, s.n + n, 1 << 16); X2 = np.zeros((cap, 2 ** N, N)); I2 = np.zeros((cap, N + 1), np.int32); X2[:s.n] = s.X[:s.n]; I2[:s.n] = s.I[:s.n]; s.X, s.I, s.cap = X2, I2, cap
        s.X[s.n:s.n + n] = X; s.I[s.n:s.n + n] = np.c_[(np.full(n, s.nc),) + tuple(ii)]; hid = np.arange(s.n, s.n + n); s.n += n; s.R.append(c.r); s.nc += 1
        lo = X.min(1) - 1e-9; hi = X.max(1) + 1e-9; Ks, Vs = [], []
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
    def _test(s, Y, pi, vv):
        hid = vv // KSH; y = Y[pi] - SH[vv % KSH]; X = s.X[hid]; ok = ((y >= X.min(1) - 1e-9) & (y <= X.max(1) + 1e-9)).all(1); pi, hid, y, X = pi[ok], hid[ok], y[ok], X[ok]
        if not len(pi): return np.zeros(0, int), np.zeros(0, int), np.zeros((0, N))
        X5 = X.reshape((-1,) + (2,) * N + (N,)); sc = np.full((len(pi), N), .5); dw = np.broadcast_to(np.array([-1., 1.]), (len(pi), 2))
        def wts(sc): return [np.stack([1 - sc[:, k], sc[:, k]], 1) for k in range(N)]
        for _ in range(8):
            w = wts(sc); F = np.einsum(EIN, X5, *w) - y
            J = np.stack([np.einsum(EIN, X5, *[dw if kk == j else w[kk] for kk in range(N)]) for j in range(N)], 2)
            try: d = np.linalg.solve(J, F[..., None])[..., 0]
            except np.linalg.LinAlgError: d = np.linalg.solve(J + 1e-12 * np.eye(N), F[..., None])[..., 0]
            sc = sc - d
        F = np.einsum(EIN, X5, *wts(sc)) - y; tol = 1e-7
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
def growN(p, u, idx, rm, tm):
    """клетка = ящик индексов; направление (ось k ±, F, B) растёт, пока: в поле, изгиб в эллипсоиде ≤ DELTA, не барьер; упёрлось в соседа (грань покрыта > FRAC) — добираем OVH·h и стоп."""
    e = basis(p, u); S = np.linspace(-rm, rm, KF); k0 = KF // 2; nmax = int(tm / DTN + 1e-9); lat = np.stack(np.meshgrid(*([S] * m_), indexing='ij'), -1); base = p + lat @ e; rows = {0: base}; Wt = {0: np.zeros((N, N))}
    for sg in (1, -1):
        y = base; yc = p.copy(); W = np.zeros((N, N))
        for i in range(1, nmax + 1):
            y = step(y, u, float(sg)); yc = step(yc, u, float(sg)); W = wstep(W, jac(yc, u), Bq(yc), DTN, sg); rows[sg * i] = y; Wt[sg * i] = W * (i * DTN)
            if not inbox(yc): break
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
                if nk < 0 or nk >= KF: act[d] = False; continue
                face = faceof(d, nk); nlo = list(klo); nhi = list(khi); nlo[k] = min(klo[k], nk); nhi[k] = max(khi[k], nk)
                if not inbox(face).all() or bend(nlo, nhi, ilo, ihi) > DELTA or (GOALB and nearb(face).any()): act[d] = False; continue
                klo, khi = nlo, nhi
                if d in extra:
                    extra[d] -= 1
                    if extra[d] <= 0: act[d] = False
                elif (idx.covered(face) | ~inbox(face)).mean() > FRAC: extra[d] = int(np.ceil(OVH * (khi[k] - klo[k]) / (M - 1)))
            else:
                i = ihi + 1 if d[0] == 'F' else ilo - 1
                if i < imin or i > imax: act[d] = False; continue
                face = R[(i - imin,) + tuple(slice(a, b + 1) for a, b in zip(klo, khi))].reshape(-1, N)
                if not inbox(face).all() or bend(klo, khi, min(ilo, i), max(ihi, i)) > DELTA or (GOALB and nearb(face).any()): act[d] = False; continue
                if d[0] == 'F': ihi = i
                else: ilo = i
                if d in extra: act[d] = False
                elif (idx.covered(face) | ~inbox(face)).mean() > FRAC: extra[d] = 1
    r = np.array([(S[b] - S[a]) / 2 for a, b in zip(klo, khi)]); near = np.linalg.norm(wrapy(p)) < GNEAR
    if ihi - ilo < (1 if near else MINROWS) or r.min() < (.02 if near else RMIN): return None
    cen = p + sum(e[k] * (S[klo[k]] + S[khi[k]]) / 2 for k in range(m_)); c = Cell(cen, u, r / (1 + HALO), e); c.nf, c.nb = ihi, -ilo; return c
def glimits(p, u):
    d = float(np.linalg.norm(wrapy(p))); return float(np.clip(GLIM * d, .04, RMAX)), float(np.clip(2 * d, .3, TMAX))
LIMITS = glimits if GLIM > 0 else None
def rand_seed(rng): return np.array([rng.uniform(-PERIOD[k] / 2, PERIOD[k] / 2) if PER[k] else rng.uniform(-XLV[k], XLV[k]) for k in range(N)])
def build_layer(u, rng, log=None):
    cells = []; fails = 0; idx = HexIdx(); queue = []; bar = tqdm(total=NFAIL, desc='layer u=%s' % (u,), mininterval=TQ, leave=False); ctr = (M // 2,) * m_
    while fails < NFAIL and len(cells) < MAXC:
        if len(cells) % 10 == 0: bar.n = fails; bar.set_postfix(cells=len(cells), queue=len(queue)); bar.refresh()
        p = queue.pop(0) if queue else rand_seed(rng)
        if not inbox(p): continue
        p = wrapy(p)
        if ingoal(p): continue
        if idx.covered(p[None])[0]: fails += 0 if queue else 1; continue
        rm, tm = LIMITS(p, u) if LIMITS else (RMAX, TMAX); c = growN(p, u, idx, rm, tm)
        if c is None: fails += 0 if queue else 1; continue
        fails = 0; c.build(); cells.append(c); idx.add(c); g0, g1 = c.G[-1], c.G[0]; yf = g0[ctr]; yb = g1[ctr]
        for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): yf = step(yf, u); yb = step(yb, u, -1.)
        queue += [yf, yb]
        for e0 in (c.c, g0[ctr], g1[ctr]):
            for k in range(m_): queue += [e0 + 1.9 * c.r[k] * c.e[k], e0 - 1.9 * c.r[k] * c.e[k]]
        if log: log(u, cells)
    bar.close(); return cells, idx
def _layer(a): return build_layer(a[0], np.random.default_rng(a[1]), None)[0]
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
        Y = wrapy(Y); pi, hid, sc = s.qx.query(Y)
        if not len(pi): return np.zeros(0, int), np.zeros((0, 2 ** N), np.int64), np.zeros((0, 2 ** N), np.float32)
        ii = s.qx.I[hid].astype(np.int64); cid = ii[:, 0]; stv = np.array([M ** (m_ - k) for k in range(N)]); base = s.O[cid] + (ii[:, 1:] * stv).sum(1); off = VOFF @ stv
        W = np.prod(np.where(VOFF[None] == 1, sc[:, None, :], 1 - sc[:, None, :]), 2)
        return pi, base[:, None] + off[None], W.astype(np.float32)
    @staticmethod
    def interp(W, VI):
        v = (W * VI).sum(1); bad = ((W > 1e-6) & (VI >= BIG / 2)).any(1); v[bad] = BIG; return v
    def vstar(s, Y):
        I, IDX, W = s.stencils(Y); out = np.full(len(Y), BIG)
        if len(I): np.minimum.at(out, I, s.interp(W, s.V[IDX]))
        out[ingoal(Y)] = 0.; return out
    def tgoal(s, Y, u, n=8):
        tg = np.full(len(Y), np.inf); y = Y
        for i in range(1, n + 1): y = rk4(y, u, DTN / n, 1); tg = np.where(np.isinf(tg) & ingoal(y), DTN * i / n, tg)
        return tg
    def solve(s, it=20000):
        I_, IDX_, W_ = [], [], []; Vg = np.full(s.N, np.inf)
        for u in US:
            Vg = np.minimum(Vg, s.tgoal(s.P, u)); a, b, c_ = s.stencils(step(s.P, u)); print('stencils built for u =', u, flush=True); I_.append(a); IDX_.append(b.astype(np.int32 if s.N < 2 ** 31 else np.int64)); W_.append(c_)
        I = np.concatenate(I_); IDX = np.concatenate(IDX_); W = np.concatenate(W_); o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]
        st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; V = np.minimum(s.V, Vg)
        bar = tqdm(total=it, desc='solve', mininterval=TQ, leave=False)
        for n in range(it):
            bar.update(1); val = DTN + s.interp(W, V[IDX]); new = V.copy(); new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st)); new[s.goal] = 0.
            d = np.max(np.abs(new - V)); V = new
            if d < 1e-9: break
        s.V = V; s.n_it = n; s.edges = len(I); return s
    def rollout(s, Q, tmax=40.):
        Y = wrapy(Q); n = len(Y); T = np.zeros(n); done = ingoal(Y); sw = np.zeros(n, int); pk = np.full(n, -1); path = [Y.copy()]
        for _ in range(int(tmax / DTN)):
            if done.all(): break
            tgs = np.stack([s.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([s.vstar(step(Y, u)) for u in US], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
            stuck = J.min(1) >= BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
            for ki, ui in enumerate(US):
                m = act & (k == ki)
                if not m.any(): continue
                hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
                for i in range(1, 9): y_i = rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
                Yn[m] = y8
            sw += act & (pk >= 0) & (k != pk); pk = np.where(act, k, pk); Y = wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= ingoal(Y) | stuck; path.append(Y.copy())
        T[~ingoal(Y)] = np.inf; return T, sw, np.array(path)
HERE = os.path.dirname(os.path.abspath(__file__)); REF = os.path.join(HERE, '../../../v5chain/reports/research')
def starts_ref():
    """(Q, эталон T)."""
    if SYS == 'dd': rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]; return Q, np.load(os.path.join(REF, 'dd_ref_60.npy'))
    if SYS == 'manip':
        rng = np.random.default_rng(0)
        for _ in range(32): rng.uniform(-1, 1, 4)
        Q = np.array([[*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)] for _ in range(8)]); ref = np.full(8, np.nan)
        for i in (0, 1, 3, 5): ref[i] = json.load(open(os.path.join(REF, 'manip_bruteforce_%d.json' % i)))[str(i)]['T']
        return Q, ref
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1); fb = os.path.join(REF, 'pend_ref_best_u%s.npy' % E('UM', '0.3'))
    return Q, np.load(fb if os.path.exists(fb) else os.path.join(HERE, 'pend_ref_T.npy'))
if __name__ == '__main__':
    t0 = time.time(); A = Atlas(); tb = time.time() - t0
    print('построено', [len(l) for l in A.layers], 'узлов', A.N, round(tb), 'с', flush=True); A.solve(); Q, ref = starts_ref()
    T, sw, _ = A.rollout(Q); fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]
    if E('DUMP'): import pickle; pickle.dump(dict(layers=A.layers, V=A.V, Q=Q, T=T, ref=ref), open(E('DUMP'), 'wb'))
    print(json.dumps(dict(SYS=SYS, DTN=DTN, RMAX=RMAX, TMAX=TMAX, DELTA=DELTA, cells=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), big_nodes=round(float((A.V >= BIG / 2).mean()), 3), reach=round(float(np.isfinite(T).mean()), 3),
                          T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)) if fz.any() else None, sec_build=round(tb, 1), sec=round(time.time() - t0, 1))), flush=True)
