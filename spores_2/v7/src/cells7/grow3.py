"""w21 (PLAN п.17б): grow3 — растущие клетки-споры для дифдрайва (x, y, θ), ромб U: 4 вершины (v, ω) = (±1, 0), (0, ±1). По одному атласу на управление.
Клетка = ящик индексов [k1lo,k1hi]×[k2lo,k2hi] × строки [ilo,ihi] на мелкой сетке KF×KF сечения (плоскость ⟂ f(p), оси e1, e2 — правые сингулярные векторы P·J·P) × время;
рост в 6 сторон по правилам grow2 (v7/grow_cells2d.py): изгиб среза (отклонение от билинейной заплатки по 4 углам) в эллипсоиде достижимости a²·T·W ≤ DELTA, наложение на соседа OVH·h.
Узлы M×M на строку (с гало), шестигранники между строками; поиск — индекс по рамкам шестигранников + трилинейная обратная карта Ньютоном 3×3.
Цена: V(узел) = min_u [Δt + V*(φ_u(узел, Δt))], V*(точка) = min по шестигранникам трилинейно; агент каждые DTN берёт argmin по 4 управлениям."""
import numpy as np, os, sys, json, time, itertools
from tqdm import tqdm
E = os.environ.get
M = 5; BIG = 1e3; HALO = .1; PER = 2 * np.pi; EPSJ = 1e-5
DTN = float(E('DTN', .1)); RMAX = float(E('RMAX', .3 if int(E('OBST', 0)) else .5)); TMAX = float(E('TMAX', 3.)); RHO = float(E('RHO', .05)); DELTA = float(E('DELTA', .03)); KF = int(E('KF', 21)); RS = int(E('RS', 3))
OVH = float(E('OVH', .5)); FRAC = float(E('FRAC', .5)); RMIN = float(E('RMIN', .02)); MINROWS = int(E('MINROWS', 1)); GNEAR = float(E('GNEAR', .7)); NFAIL = int(E('NFAIL', 400)); XL = float(E('XL', 2.5)); QB = float(E('QB', .25))
US = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.)); SH = (0., PER, -PER)
def f(y, u): th = y[..., 2]; return np.stack([u[0] * np.cos(th), u[0] * np.sin(th), u[1] + 0 * th], -1)
def wrap(th): return (th + np.pi) % PER - np.pi
def wrapy(y): y = np.array(y, float); y[..., 2] = wrap(y[..., 2]); return y
def rk4(y, u, h, n):
    for _ in range(n):
        k1 = f(y, u); k2 = f(y + h / 2 * k1, u); k3 = f(y + h / 2 * k2, u); k4 = f(y + h * k3, u); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
def step(y, u, sg=1.): return rk4(y, u, sg * DTN / 2, 2)
def ingoal(y): return (np.abs(y[..., 0]) <= RHO + 1e-9) & (np.abs(y[..., 1]) <= RHO + 1e-9) & (np.abs(wrap(y[..., 2])) <= RHO + 1e-9)
OBST = int(E('OBST', 0)); DISCS = [(1., .3, .45), (-.8, -.9, .4)] if OBST else []               # 17г: two discs, same as v7/reports/bdd/obst_ref.py (reference dd_obst_ref_KB60_NG21.npy)
def dobs(y): return np.min([np.hypot(y[..., 0] - cx, y[..., 1] - cy) - r for cx, cy, r in DISCS], 0) if DISCS else np.full(np.shape(y[..., 0]), 9.)   # signed distance to the nearest disc
def inbox(y): return (np.abs(y[..., 0]) <= XL) & (np.abs(y[..., 1]) <= XL) & (dobs(y) > 0)
GM = float(E('GM', 1.))
def inbox_g(y): return (np.abs(y[..., 0]) <= XL + GM) & (np.abs(y[..., 1]) <= XL + GM) & (dobs(y) > 0)   # r17 GM: cells may grow GM beyond the field (seeds stay inside); the field border no longer cuts cell length
def jac(x, u): return np.stack([(f(x + EPSJ * e, u) - f(x - EPSJ * e, u)) / (2 * EPSJ) for e in np.eye(3)], 1)
def bbt(th): c, s = np.cos(th), np.sin(th); return np.array([[c * c, c * s, 0], [c * s, s * s, 0], [0, 0, 1.]])            # B = ∂f/∂(v, ω), |δu| ≤ 1 в обоих каналах
def wstep(W, A, Bq, h, sg):
    A = sg * A; Ph = np.eye(3) + h * A + h * h / 2 * A @ A
    return Ph @ W @ Ph.T + h * Bq + h * h / 2 * (A @ Bq + Bq @ A.T) + h ** 3 / 3 * A @ Bq @ A.T
def basis(p, u):
    """оси сечения: правые сингулярные векторы P·J·P; J = 0 (повороты) — проекции x, y."""
    f0 = f(p, u); fh = f0 / np.linalg.norm(f0); P = np.eye(3) - np.outer(fh, fh); S, Vt = np.linalg.svd(P @ jac(p, u) @ P)[1:]
    if S[0] < 1e-9:
        e1 = P @ np.array([1., 0, 0])
        if np.linalg.norm(e1) < 1e-6: e1 = P @ np.array([0, 1., 0])
    else: e1 = Vt[0]
    e1 = e1 / np.linalg.norm(e1); e2 = np.cross(fh, e1); return e1, e2 / np.linalg.norm(e2)
class Cell:
    def __init__(s, c, u, r1, r2, e1, e2): s.c, s.u, s.r1, s.r2, s.e1, s.e2 = np.array(c, float), u, r1, r2, e1, e2; s.nb = 0; s.nf = 0; s.m = M
    def build(s):
        """сетка узлов G (nt, M, M, 3): заплатка с гало, пронесённая потоком на nb шагов назад и nf вперёд."""
        a = np.linspace(-(1 + HALO) * s.r1, (1 + HALO) * s.r1, M); b = np.linspace(-(1 + HALO) * s.r2, (1 + HALO) * s.r2, M); seg = s.c + a[:, None, None] * s.e1 + b[None, :, None] * s.e2; fw = [seg]; bw = []; y = seg
        for _ in range(s.nf): fw.append(step(fw[-1], s.u))
        for _ in range(s.nb): y = step(y, s.u, -1.); bw.append(y)
        s.G = np.array(bw[::-1] + fw)
        if RS > 1: nt = len(s.G); keep = sorted(set(range(0, nt, RS)) | {nt - 1}); s.G = s.G[keep]                 # r17: rows every RS·DTN (dd flow is linear in t at fixed u — geometry exact)
class HexIdx:
    """индекс шестигранников: ячейки QB³ (θ с копиями ±2π), запрос — все точки сразу; трилинейная обратная карта Ньютоном."""
    def __init__(s): s.n = 0; s.cap = 0; s.X = np.zeros((0, 8, 3)); s.XL = np.zeros((0, 3)); s.XH = np.zeros((0, 3)); s.I = np.zeros((0, 4), np.int32); s.bins = {}; s.R1 = []; s.R2 = []; s.nc = 0
    def add(s, c):
        g = c.G; nt, m = g.shape[0], g.shape[1]
        X = np.stack([g[di:nt - 1 + di, d1:m - 1 + d1, d2:m - 1 + d2] for di in (0, 1) for d1 in (0, 1) for d2 in (0, 1)], 3).reshape(-1, 8, 3)
        it, j1, j2 = [a.ravel() for a in np.indices((nt - 1, m - 1, m - 1))]; n = len(X)
        if s.n + n > s.cap:
            cap = max(2 * s.cap, s.n + n, 1 << 16); X2 = np.zeros((cap, 8, 3)); I2 = np.zeros((cap, 4), np.int32); X2[:s.n] = s.X[:s.n]; I2[:s.n] = s.I[:s.n]; L2 = np.zeros((cap, 3)); H2 = np.zeros((cap, 3)); L2[:s.n] = s.XL[:s.n]; H2[:s.n] = s.XH[:s.n]; s.X, s.I, s.cap, s.XL, s.XH = X2, I2, cap, L2, H2
        s.X[s.n:s.n + n] = X; s.I[s.n:s.n + n] = np.c_[np.full(n, s.nc), it, j1, j2]; hid = np.arange(s.n, s.n + n); s.n += n; s.R1.append(c.r1); s.R2.append(c.r2); s.nc += 1
        lo = X.min(1) - 1e-9; hi = X.max(1) + 1e-9; s.XL[s.n - n:s.n] = lo; s.XH[s.n - n:s.n] = hi; Ks, Vs = [], []
        for si, sh in enumerate(SH):
            l, h = lo.copy(), hi.copy(); l[:, 2] += sh; h[:, 2] += sh; v = (h[:, 2] >= -np.pi - QB) & (l[:, 2] <= np.pi + QB)
            if not v.any(): continue
            i0 = np.floor(l[v] / QB).astype(np.int64); i1 = np.floor(h[v] / QB).astype(np.int64); hv = hid[v]; sp = (i1 - i0).max(0)
            for dx in range(sp[0] + 1):
                for dy in range(sp[1] + 1):
                    for dz in range(sp[2] + 1):
                        k = (i0[:, 0] + dx <= i1[:, 0]) & (i0[:, 1] + dy <= i1[:, 1]) & (i0[:, 2] + dz <= i1[:, 2])
                        if k.any(): Ks.append(((i0[k, 0] + dx + 512) * 1024 + (i0[k, 1] + dy + 512)) * 1024 + (i0[k, 2] + dz + 512)); Vs.append(hv[k] * 3 + si)
        if not Ks: return
        K = np.concatenate(Ks); V = np.concatenate(Vs); o = np.argsort(K, kind='stable'); K, V = K[o], V[o]; uk, st = np.unique(K, return_index=True); st = np.r_[st, len(K)]
        for k_, a_, b_ in zip(uk.tolist(), st[:-1], st[1:]): s.bins.setdefault(k_, []).append(V[a_:b_])
    def query(s, Y):
        """пары (точка, шестигранник), где точка внутри: pi, hid, t, p, q (координаты в шестиграннике: время, поперёк 1, поперёк 2)."""
        z = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0))
        if s.n == 0 or not len(Y): return z
        ky = np.floor(Y / QB).astype(np.int64); kk = ((ky[:, 0] + 512) * 1024 + (ky[:, 1] + 512)) * 1024 + (ky[:, 2] + 512); o = np.argsort(kk, kind='stable'); uk, st = np.unique(kk[o], return_index=True); st = np.r_[st, len(o)]
        PI, VV, out, cnt = [], [], [], 0
        for k_, a_, b_ in zip(uk.tolist(), st[:-1], st[1:]):
            lst = s.bins.get(k_)
            if not lst: continue
            if len(lst) > 1: lst = [np.concatenate(lst)]; s.bins[k_] = lst
            arr = lst[0]; pts = o[a_:b_]; PI.append(np.repeat(pts, len(arr))); VV.append(np.tile(arr, len(pts))); cnt += len(pts) * len(arr)
            if cnt > 1_500_000: out.append(s._test(Y, np.concatenate(PI), np.concatenate(VV))); PI, VV, cnt = [], [], 0
        if PI: out.append(s._test(Y, np.concatenate(PI), np.concatenate(VV)))
        if not out: return z
        return [np.concatenate(a) for a in zip(*out)]
    def _test(s, Y, pi, vv):
        hid = vv // 3; sh = np.array(SH)[vv % 3]; y = Y[pi].copy(); y[:, 2] -= sh; ok = ((y >= s.XL[hid]) & (y <= s.XH[hid])).all(1); pi, hid, y = pi[ok], hid[ok], y[ok]; X = s.X[hid]   # w22: precomputed per-hex bounds (the gather+min/max of all 8 corners was ~60% of build)
        if not len(pi): return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0)
        X5 = X.reshape(-1, 2, 2, 2, 3); t = np.full(len(pi), .5); p = t.copy(); q = t.copy()
        for _ in range(6):
            Yq = (1 - q)[:, None, None, None] * X5[:, :, :, 0] + q[:, None, None, None] * X5[:, :, :, 1]; Z = (1 - p)[:, None, None] * Yq[:, :, 0] + p[:, None, None] * Yq[:, :, 1]
            F = (1 - t)[:, None] * Z[:, 0] + t[:, None] * Z[:, 1] - y; Ja = Z[:, 1] - Z[:, 0]; Jb = (1 - t)[:, None] * (Yq[:, 0, 1] - Yq[:, 0, 0]) + t[:, None] * (Yq[:, 1, 1] - Yq[:, 1, 0])
            dq = X5[:, :, :, 1] - X5[:, :, :, 0]; dqp = (1 - p)[:, None, None] * dq[:, :, 0] + p[:, None, None] * dq[:, :, 1]; Jc = (1 - t)[:, None] * dqp[:, 0] + t[:, None] * dqp[:, 1]
            JbC = np.cross(Jb, Jc); det = (Ja * JbC).sum(1); det = np.where(np.abs(det) < 1e-18, 1e-18, det)
            t = t - (F * JbC).sum(1) / det; p = p - (Ja * np.cross(F, Jc)).sum(1) / det; q = q - (Ja * np.cross(Jb, F)).sum(1) / det
        Yq = (1 - q)[:, None, None, None] * X5[:, :, :, 0] + q[:, None, None, None] * X5[:, :, :, 1]; Z = (1 - p)[:, None, None] * Yq[:, :, 0] + p[:, None, None] * Yq[:, :, 1]
        F = (1 - t)[:, None] * Z[:, 0] + t[:, None] * Z[:, 1] - y; tol = 1e-7
        k = (np.linalg.norm(F, axis=1) < 1e-7) & (t >= -tol) & (t <= 1 + tol) & (p >= -tol) & (p <= 1 + tol) & (q >= -tol) & (q <= 1 + tol)
        return pi[k], hid[k], np.clip(t[k], 0, 1), np.clip(p[k], 0, 1), np.clip(q[k], 0, 1)
    def covered(s, Y):
        Y = wrapy(np.atleast_2d(Y)); m = np.zeros(len(Y), bool); pi, hid, t, p, q = s.query(Y)
        if len(pi):
            R1 = np.array(s.R1); R2 = np.array(s.R2); cid = s.I[hid, 0]; j1 = s.I[hid, 2]; j2 = s.I[hid, 3]; w = 2 * (1 + HALO) / (M - 1)
            s1 = -(1 + HALO) * R1[cid] + (j1 + p) * w * R1[cid]; s2 = -(1 + HALO) * R2[cid] + (j2 + q) * w * R2[cid]; core = (np.abs(s1) <= R1[cid] + 1e-9) & (np.abs(s2) <= R2[cid] + 1e-9); m[pi[core]] = True
        return m
BARRIER = None; BEPS = float(E('BEPS', .01)); GOALB = int(E('GOALB', 1))
if GOALB:                                                                                       # goal box surface is a wall from the start (cf. 2D GOALB): cells end at it, so their last nodes reach the goal within one step
    from scipy.spatial import cKDTree
    _e = np.linspace(-RHO, RHO, 21); _a, _b = np.meshgrid(_e, _e); _a, _b = _a.ravel(), _b.ravel(); _o = np.zeros_like(_a)
    _B = [np.concatenate([np.c_[_a, _b, _o + s_ * RHO][:, p_] for s_ in (-1, 1) for p_ in ([0, 1, 2], [0, 2, 1], [2, 0, 1])])]
    for cx, cy, r in DISCS:                                                                         # disc walls (17g): cylinder surface x, y on the circle, any θ
        _t = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / .005), endpoint=False); _z = np.arange(-np.pi, np.pi, .015); _B.append(np.c_[np.repeat(cx + r * np.cos(_t), len(_z)), np.repeat(cy + r * np.sin(_t), len(_z)), np.tile(_z, len(_t))])
    BARRIER = cKDTree(np.concatenate(_B))
def nearb(P): return np.zeros(len(P), bool) if BARRIER is None else BARRIER.query(wrapy(P), distance_upper_bound=BEPS)[0] < BEPS
def grow3(p, u, idx, rm, tm):
    """клетка = ящик индексов; направление (a±, b±, F, B) растёт, пока: в поле, изгиб среза в эллипсоиде ≤ DELTA, не барьер; упёрлось в соседа (грань покрыта > FRAC) — добираем OVH·h и стоп."""
    e1, e2 = basis(p, u); S = np.linspace(-rm, rm, KF); k0 = KF // 2; nmax = int(tm / DTN + 1e-9); base = p + S[:, None, None] * e1 + S[None, :, None] * e2; rows = {0: base}; Wt = {0: np.zeros((3, 3))}
    for sg in (1, -1):
        y = base; yc = p.copy(); W = np.zeros((3, 3))
        for i in range(1, nmax + 1):
            y = step(y, u, float(sg)); yc = step(yc, u, float(sg)); W = wstep(W, jac(yc, u), bbt(yc[2]), DTN, sg); rows[sg * i] = y; Wt[sg * i] = W * (i * DTN)
            if not inbox_g(yc): break
    imin, imax = min(rows), max(rows); R = np.stack([rows[i] for i in range(imin, imax + 1)])
    def bend(k1lo, k1hi, k2lo, k2hi, ilo, ihi):
        n1, n2 = k1hi - k1lo + 1, k2hi - k2lo + 1
        if n1 < 3 and n2 < 3: return 0.
        Wc = Wt[ihi] if ihi >= -ilo else Wt[ilo]; T_ = (ihi - ilo) * DTN; Wc = Wc * (T_ / max(max(ihi, -ilo) * DTN, 1e-9)); lam, Q = np.linalg.eigh(Wc); Wi = (Q / np.maximum(lam, 1e-12 * max(lam.max(), 1e-30))) @ Q.T
        Pb = R[ilo - imin:ihi - imin + 1, k1lo:k1hi + 1, k2lo:k2hi + 1]; a = np.linspace(0, 1, n1)[None, :, None, None]; b = np.linspace(0, 1, n2)[None, None, :, None]
        d = Pb - ((1 - a) * (1 - b) * Pb[:, :1, :1] + a * (1 - b) * Pb[:, -1:, :1] + (1 - a) * b * Pb[:, :1, -1:] + a * b * Pb[:, -1:, -1:])
        return float(np.sqrt(np.max(np.einsum('rabk,kl,rabl->rab', d, Wi, d))))
    k1lo, k1hi, k2lo, k2hi, ilo, ihi = k0 - 1, k0 + 1, k0 - 1, k0 + 1, 0, 0; act = {d: True for d in ('a+', 'a-', 'b+', 'b-', 'F', 'B')}; extra = {}
    while any(act.values()):
        for d in ('a+', 'a-', 'b+', 'b-', 'F', 'B'):
            if not act[d]: continue
            if d[0] in 'ab':
                if ihi - ilo < 3 and (act['F'] or act['B']): continue
                pl = d[1] == '+'
                if d[0] == 'a':
                    k = k1hi + 1 if pl else k1lo - 1
                    if k < 0 or k >= KF: act[d] = False; continue
                    face = R[ilo - imin:ihi - imin + 1, k, k2lo:k2hi + 1].reshape(-1, 3); nr = (min(k1lo, k), max(k1hi, k), k2lo, k2hi)
                else:
                    k = k2hi + 1 if pl else k2lo - 1
                    if k < 0 or k >= KF: act[d] = False; continue
                    face = R[ilo - imin:ihi - imin + 1, k1lo:k1hi + 1, k].reshape(-1, 3); nr = (k1lo, k1hi, min(k2lo, k), max(k2hi, k))
                if not inbox_g(face).all() or bend(*nr, ilo, ihi) > DELTA or (BARRIER is not None and nearb(face).any()): act[d] = False; continue
                k1lo, k1hi, k2lo, k2hi = nr
                if d in extra:
                    extra[d] -= 1
                    if extra[d] <= 0: act[d] = False
                elif (idx.covered(face) | ~inbox_g(face)).mean() > FRAC: extra[d] = int(np.ceil(OVH * ((k1hi - k1lo) if d[0] == 'a' else (k2hi - k2lo)) / (M - 1)))
            else:
                i = ihi + 1 if d == 'F' else ilo - 1
                if i < imin or i > imax: act[d] = False; continue
                face = R[i - imin, k1lo:k1hi + 1, k2lo:k2hi + 1].reshape(-1, 3)
                if not inbox_g(face).all() or bend(k1lo, k1hi, k2lo, k2hi, min(ilo, i), max(ihi, i)) > DELTA or (BARRIER is not None and nearb(face).any()): act[d] = False; continue
                if d == 'F': ihi = i
                else: ilo = i
                if d in extra: act[d] = False
                elif (idx.covered(face) | ~inbox_g(face)).mean() > FRAC: extra[d] = 1
    r1 = (S[k1hi] - S[k1lo]) / 2; r2 = (S[k2hi] - S[k2lo]) / 2
    near = np.linalg.norm(np.r_[p[:2], wrap(p[2])]) < GNEAR                                   # near the goal slivers are kept (walls make cells small there, holes would break V propagation)
    if ihi - ilo < (1 if near else MINROWS) or min(r1, r2) < (.02 if near else RMIN): return None   # MINROWS/RMIN: refuse sliver cells, overlap (OVH) covers the gaps instead
    c = Cell(p + e1 * (S[k1lo] + S[k1hi]) / 2 + e2 * (S[k2lo] + S[k2hi]) / 2, u, r1 / (1 + HALO), r2 / (1 + HALO), e1, e2); c.nf, c.nb = ihi, -ilo; return c
LRTA = float(E('LRTA', .03)); VF0 = float(E('VF0', 1.5)); VFR = float(E('VFR', 1.0)); NOPR = int(E('NOPR', 6)); MACRO = int(E("MACRO", 0)); NOPD = float(E("NOPD", .02)); VF = float(E('VF', 5.)); NA = int(E('NA', 4))                                              # finish by shooting (research-17): at V* <= VF the agent switches to <= NA rhombus-vertex arcs (cure for chattering at |y|~.17 near the goal)
GLIM = float(E('GLIM', .25))
def glimits(p, u):
    """cells near the goal must be as narrow as the goal box (else only a few nodes hit it and V does not propagate): half-width ~ GLIM·distance, length ~ 2·distance."""
    d = float(np.linalg.norm(np.r_[p[:2], wrap(p[2])])); return float(np.clip(GLIM * d, .04, RMAX)), float(np.clip(2 * d, .3, TMAX))
LIMITS = glimits if GLIM > 0 else None
def build_layer(u, rng, log=None):
    cells = []; fails = 0; idx = HexIdx(); queue = []; bar = tqdm(total=NFAIL, desc='layer u=%s' % (u,), mininterval=10, leave=False)   # bar = consecutive covered random seeds, resets on each new cell
    while fails < NFAIL:
        if len(cells) % 10 == 0: bar.n = fails; bar.set_postfix(cells=len(cells), queue=len(queue)); bar.refresh()
        p = queue.pop(0) if queue else np.array([rng.uniform(-XL, XL), rng.uniform(-XL, XL), rng.uniform(-np.pi, np.pi)])
        if not inbox(p): continue
        p = wrapy(p)
        if ingoal(p): continue                                                                 # goal interior is not covered by cells (V = 0 there)
        if idx.covered(p[None])[0]: fails += 0 if queue else 1; continue
        rm, tm = LIMITS(p, u) if LIMITS else (RMAX, TMAX); c = grow3(p, u, idx, rm, tm)
        if c is None: fails += 0 if queue else 1; continue
        fails = 0; c.build(); cells.append(c); idx.add(c); g0, g1 = c.G[-1], c.G[0]; yf = g0[M // 2, M // 2]; yb = g1[M // 2, M // 2]
        for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): yf = step(yf, u); yb = step(yb, u, -1.)
        queue += [yf, yb]
        for e0, k in ((c.c, 1.9), (g0[M // 2, M // 2], 1.9), (g1[M // 2, M // 2], 1.9)): queue += [e0 + k * c.r1 * c.e1, e0 - k * c.r1 * c.e1, e0 + k * c.r2 * c.e2, e0 - k * c.r2 * c.e2]
        if log: log(u, cells)
    bar.close(); return cells, idx
def _layer(a): return build_layer(a[0], np.random.default_rng(a[1]), None)[0]
def arc(y, u, t):
    x, yy, th = y; v, w = u
    if w == 0: return np.array([x + v * t * np.cos(th), yy + v * t * np.sin(th), th])
    return np.array([x, yy, th + w * t])
TOPS = [tp for n in range(1, NA + 1) for tp in itertools.product(range(4), repeat=n) if all(tp[i] != tp[i + 1] for i in range(n - 1))]
def _free(y, tp, d, n=30):
    """arcs of the shooting finish stay outside the discs (n points per arc)"""
    z = np.array(y, float)
    for k, dt in zip(tp, d):
        pts = np.array([arc(z, US[k], dt * a) for a in np.linspace(0, 1, n)])
        if (dobs(pts) <= 0).any(): return False
        z = arc(z, US[k], dt)
    return True
def _key(y): return (int(round(y[0] / .06)), int(round(y[1] / .06)), int(round(wrap(y[2]) / .2)))
def shoot(y, tmax):
    """best shooting finish from y: arcs of the 4 rhombus vertices in closed form, durations by SLSQP, end inside the goal box; returns (time, topology) or (inf, None)."""
    from scipy.optimize import minimize
    best = (np.inf, None); R = RHO * .9
    for tp in TOPS:
        def end(d):
            z = np.array(y, float)
            for k, dt in zip(tp, d): z = arc(z, US[k], dt)
            return np.r_[z[:2], wrap(z[2])]
        cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
        for d0 in (np.full(len(tp), tmax / (2 * len(tp))), np.full(len(tp), tmax / len(tp))):
            r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * len(tp), constraints=cons, options=dict(maxiter=100, ftol=1e-9))
            if r.success and np.all(np.abs(end(r.x)) <= RHO) and r.x.sum() < best[0] and (not DISCS or _free(y, tp, r.x)): best = (r.x.sum(), tp)
    return best
class Atlas:
    def __init__(s, seed=0, log=None):
        s.layers = []; s.idx = []
        if int(E('PAR', 1)):                                                                                  # one process per control: layers are independent
            from multiprocessing import Pool
            with Pool(len(US)) as pool: s.layers = pool.map(_layer, [(u, seed + k) for k, u in enumerate(US)])
        else: s.layers = [_layer((u, seed + k)) for k, u in enumerate(US)]
        s.finish()
    def finish(s):
        s.cells = [c for l in s.layers for c in l]; N = 0
        for c in s.cells: c.o = N; N += c.G.shape[0] * M * M
        s.P = np.concatenate([c.G.reshape(-1, 3) for c in s.cells]); s.N = N; s.goal = ingoal(s.P); s.V = np.full(N, BIG); s.V[s.goal] = 0.
    def stencils(s, Y):
        if not hasattr(s, 'qx'):
            s.qx = HexIdx()
            for c in s.cells: s.qx.add(c)
            s.O = np.array([c.o for c in s.cells])
        Y = wrapy(Y); pi, hid, t, p, q = s.qx.query(Y)
        if not len(pi): return np.zeros(0, int), np.zeros((0, 8), np.int64), np.zeros((0, 8), np.float32)
        cid, it, j1, j2 = s.qx.I[hid].T.astype(np.int64); base = s.O[cid] + it * M * M + j1 * M + j2; off = np.array([di * M * M + d1 * M + d2 for di in (0, 1) for d1 in (0, 1) for d2 in (0, 1)])
        W = np.stack([(t if di else 1 - t) * (p if d1 else 1 - p) * (q if d2 else 1 - q) for di in (0, 1) for d1 in (0, 1) for d2 in (0, 1)], 1)
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
        for u in US: Vg = np.minimum(Vg, s.tgoal(s.P, u)); a, b, c_ = s.stencils(step(s.P, u)); print('stencils built for u =', u, flush=True); I_.append(a); IDX_.append(b); W_.append(c_)
        I = np.concatenate(I_); IDX = np.concatenate(IDX_); W = np.concatenate(W_); o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]
        sf = IDX == I[:, None]; ws = (W * sf).sum(1); W = np.where(sf, 0, W).astype(np.float32); den = 1. - ws; den[den < 1e-6] = np.nan   # r17: self-loop (RS > 1: the step lands in its own hex) solved exactly: V = (DTN + sum_{j!=i} w_j V_j)/(1 - w_ii)
        st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; V = np.minimum(s.V, Vg)
        bar = tqdm(total=it, desc='solve', mininterval=10, leave=False)
        for n in range(it):
            bar.update(1); val = np.nan_to_num((DTN + s.interp(W, V[IDX])) / den, nan=BIG); val[val > BIG] = BIG; new = V.copy(); new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st)); new[s.goal] = 0.
            d = np.max(np.abs(new - V)); V = new
            if d < 1e-9: break
        s.V = V; s.n_it = n; s.edges = len(I); return s
    def rollout(s, Q, tmax=40., vf=None):
        vf = VF if vf is None else vf
        Y = wrapy(Q); n = len(Y); T = np.zeros(n); done = ingoal(Y); sw = np.zeros(n, int); vbest = np.full(n, np.inf); lr = np.zeros(n, bool); vis = [dict() for _ in range(n)]; nopr = np.zeros(n, int); pk = np.full(n, -1); path = [Y.copy()]; fin = np.zeros(n, bool)
        for _ in range(int(tmax / DTN)):
            if done.all(): break
            if vf > 0:
                v = s.vstar(Y)
                for i in np.flatnonzero(~done & (v <= vf)):
                    tf, tp = shoot(Y[i], 2.5 * max(v[i], .2))
                    if tp is not None and (v[i] <= VF0 or tf <= VFR * v[i] + .05): T[i] += tf; done[i] = True; fin[i] = True   # w22: far from the goal take the shot only if it is not worse than the atlas estimate (SLSQP local optima around discs)
                if done.all(): break
            tgs = np.stack([s.tgoal(Y, u) for u in US], 1); Ys = [step(Y, u) for u in US]; Jv = DTN + np.stack([s.vstar(y_) for y_ in Ys], 1)
            if LRTA:                                                                                    # w22 (LRTA*): once an agent has stalled, every visit to a coarse cell adds LRTA to the cost of stepping into it — breaks limit cycles beside walls
                for i in np.flatnonzero(lr & ~done):
                    for ki in range(len(US)): Jv[i, ki] += LRTA * vis[i].get(_key(Ys[ki][i]), 0)
                    vis[i][_key(Y[i])] = vis[i].get(_key(Y[i]), 0) + 1
            J = np.minimum(tgs, Jv); k = J.argmin(1); tg = tgs[np.arange(n), k]
            stuck = J.min(1) >= BIG / 2
            if vf > 0:                                                                                  # all 4 steps land in holes (e.g. beside a wall): finish by shooting from here, whatever V* is
                vc = s.vstar(Y); nopr = np.where(vc <= vbest - NOPD, 0, nopr + 1); vbest = np.minimum(vbest, vc)
                for i in np.flatnonzero(~done & (nopr >= NOPR)):                                          # w22: V* stopped decreasing (limit cycle beside a wall, start 51): try the shoot finish, keep going if it fails
                    nopr[i] = 0; lr[i] = True; tf, tp = shoot(Y[i], 2.5 * max(float(vc[i]) if vc[i] < BIG / 2 else 4., .2))
                    if tp is not None: T[i] += tf; done[i] = True; fin[i] = True
                    elif MACRO:                                                                                  # w22: escape — one long constant-control arc (4..16 DTN) with the best t + V*(end), outside the discs
                        best = (np.inf, None)
                        for ki, ui in enumerate(US):
                            for m_ in (2, 4, 8, 16):
                                z = arc(Y[i], ui, m_ * DTN); z = np.r_[z[:2], wrap(z[2])]; c = m_ * DTN + float(s.vstar(z[None])[0])
                                if c < best[0] and inbox(z) and _free(Y[i], (ki,), (m_ * DTN,)): best = (c, (ki, m_, z))
                        if best[1] is not None: k_, m_, z = best[1]; Y[i] = z; T[i] += m_ * DTN; vbest[i] = np.inf
                for i in np.flatnonzero(~done & stuck):
                    tf, tp = shoot(Y[i], 2.5 * max(float(s.vstar(Y[i:i + 1])[0]) if s.vstar(Y[i:i + 1])[0] < BIG / 2 else 4., .2))
                    if tp is not None: T[i] += tf; done[i] = True; fin[i] = True
                stuck &= ~done
            act = ~done & ~stuck; Yn = Y.copy()
            for ki, ui in enumerate(US):
                m = act & (k == ki)
                if not m.any(): continue
                hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
                for i in range(1, 9): y_i = rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
                Yn[m] = y8
            sw += act & (pk >= 0) & (k != pk); pk = np.where(act, k, pk); Y = wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= ingoal(Y) | stuck; path.append(Y.copy())
        T[~(ingoal(Y) | fin)] = np.inf; s.n_fin = int(fin.sum()); return T, sw, np.array(path)
def starts_ref():
    if OBST: o = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../v5chain/reports/research/dd_obst_ref_KB60_NG21.npy')); return o[:, :3], o[:, 3]
    rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]
    return Q, np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../v5chain/reports/research/dd_ref_60.npy'))
if __name__ == '__main__':
    t0 = time.time(); A = Atlas(); tb = time.time() - t0
    print('построено', [len(l) for l in A.layers], 'узлов', A.N, round(tb), 'с', flush=True); A.solve(); Q, ref = starts_ref(); T, sw, pth = A.rollout(Q); fz = np.isfinite(T); print('collisions (path points inside discs):', int((dobs(pth.reshape(-1, 3)) <= 0).sum()) if DISCS else 0, flush=True); r = T[fz] / ref[fz]
    if E('DUMP'): import pickle; pickle.dump(dict(layers=A.layers, V=A.V, Q=Q, T=T, ref=ref), open(E('DUMP'), 'wb'))
    print(json.dumps(dict(DTN=DTN, RMAX=RMAX, TMAX=TMAX, DELTA=DELTA, cells=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), big_nodes=round(float((A.V >= BIG / 2).mean()), 3), reach=round(float(fz.mean()), 3), VF=VF, fin=getattr(A, 'n_fin', 0),
                          T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)) if fz.any() else None, sec_build=round(tb, 1), sec=round(time.time() - t0, 1))), flush=True)
