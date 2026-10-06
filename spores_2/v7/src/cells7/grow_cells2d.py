"""w20 (PLAN п.17а): перенос в v7 из v5chain/reports/research/grow_cells2d.py (research-14: GROW2, QIdx/FASTLOC=2 по профилю). Умолчания = связка GROW2 (δ .03) + BTOUCH/BEPS .01; GOALB и CUT — в experiments/*/compute.py (умолчания GOALB=1, CUT=1 для pend / .5 для di)."""
"""research-13 (слово пользователя 2026-10-05): растущие споры-клетки для ЛЮБОЙ 2D системы (обобщение di_grow_cells.py). По одному атласу на управление.
Спора = точка + нормальный отрезок (M узлов), пронесённый потоком своего управления вперёд и назад ≤ TMAX. Посев встык, обязательные боковые (OVL) и курсовые
пересечения с соседями. Формул системы нет: поток — rk4, положение точки в клетке — поиск четырёхугольника сетки узлов + обратная билинейная карта.
Цена: V(узел) = min_u [Δt + V*(φ_u(узел, Δt))], V*(точка) = min по клеткам любого атласа билинейно; агент каждые Δt берёт argmin.
Системы: SYS=pend (φ̈ = sin φ + u, |u| ≤ .3, цель — верх ±RHO, φ периодична), SYS=di (проверка против di_grow_cells.py)."""
import numpy as np, sys, os, json, time
from tqdm import tqdm
MI = float(os.environ.get('TQDM_MI', 10))   # tqdm mininterval for log files (user rule 2026-10-06)
SYS = os.environ.get('SYS', 'pend'); M = 5; BIG = 1e3; HALO = .1
DTN = float(os.environ.get('DTN', .06)); RMAX = float(os.environ.get('RMAX', .3)); TMAX = float(os.environ.get('TMAX', 3.)); OVL = float(os.environ.get('OVL', .05)); RHO = float(os.environ.get('RHO', .1))
ADAPT = int(os.environ.get('ADAPT', 0)); TURN = np.deg2rad(float(os.environ.get('TURN', 40))); STR = float(os.environ.get('STR', 2.)); BEND = float(os.environ.get('BEND', .05)); TRV = float(os.environ.get('TRV', .5)); LMAX = float(os.environ.get('LMAX', 1.2))
CORE = int(os.environ.get('CORE', 0)); FASTLOC = int(os.environ.get('FASTLOC', 2)); JUMP = float(os.environ.get('JUMP', 0)); JMODE = os.environ.get('JMODE', 'max'); JAG = int(os.environ.get('JAG', 0)); JDIR = int(os.environ.get('JDIR', 0));   # JDIR=1: разрыв — только скачок поперёк столбцов в ОБЕИХ строках (барьер вдоль потока); JMODE drop — четырёхугольник не используется
   # JMODE near — значение узла с наибольшим весом; JAG=1 — только у агента (vstar), не в счёте V
   # JUMP > 0 (research-14): узлы четырёхугольника расходятся по V > JUMP — разрыв цены, берём max узлов (не обещать лишнего)
NORM = os.environ.get('NORM', 'gram'); DELTA = float(os.environ.get('DELTA', .03)); SEL = os.environ.get('SEL', 'stop'); EPSJ = 1e-5; STEPS = int(os.environ.get('STEPS', 0)); DN = float(os.environ.get('DN', .003)); MMAX = int(os.environ.get('MMAX', 9))   # STEPS=1: узлы поперёк m и шаг строк kt — из той же метрики (ошибка интерполяции ≤ DN); ADAPT=2 (research-14): ошибка линейной модели среза в локальной метрике
RMIN = float(os.environ.get('RMIN', .01)); FMIN = float(os.environ.get('FMIN', .2)); NFAIL = int(os.environ.get('NFAIL', 400))
if SYS == 'pend':
    UM = float(os.environ.get('UM', .3)); UB = UM;   # UM (research-15): мотор слабее — больше раскачек и сепаратрис
    US = (-UM, 0., UM); PER = 2 * np.pi; XL, WL = np.pi, float(os.environ.get('WS', 3.5))
    def f(y, u): return np.stack([y[..., 1], np.sin(y[..., 0]) + u], -1)
else:
    UB = 1.; US = (-1., 0., 1.); PER = None; XL, WL = 2.5, 2.5
    def f(y, u): return np.stack([y[..., 1], 0 * y[..., 0] + u], -1)
def wrap(x): return x if PER is None else (x + PER / 2) % PER - PER / 2
def rk4(y, u, h, n):
    for _ in range(n):
        k1 = f(y, u); k2 = f(y + h / 2 * k1, u); k3 = f(y + h / 2 * k2, u); k4 = f(y + h * k3, u); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
def step(y, u, sg=1.): return rk4(y, u, sg * DTN / 2, 2)                                        # один шаг сетки узлов (вперёд sg = 1, назад −1)
def ingoal(y): return (np.abs(wrap(y[..., 0])) <= RHO + 1e-9) & (np.abs(y[..., 1]) <= RHO + 1e-9)
def inbox(y): return (np.abs(y[..., 1]) <= WL) & ((np.abs(y[..., 0]) <= XL) if PER is None else True)
def normal(c, u): v = f(c, u); n = np.array([-v[1], v[0]]); return n / np.linalg.norm(n)
SH = (0.,) if PER is None else (0., PER, -PER)
class Cell:
    def __init__(s, c, u, r): s.c, s.u, s.r = np.array(c, float), u, r; s.n = normal(s.c, u); s.nb = 0; s.nf = 0; s.m = M; s.kt = 1; s.p0 = None; s.off = 0.
    def build(s):
        """сетка узлов G (nt, M, 2): отрезок с гало, пронесённый потоком на nb шагов назад и nf вперёд."""
        s.sn = np.linspace(-(1 + HALO) * s.r, (1 + HALO) * s.r, s.m); seg = s.c + s.sn[:, None] * s.n; fw = [seg]; bw = []
        if NORMFRONT and s.p0 is not None:                                                        # NORMFRONT: строки по нормали к потоку через центр c_i (траектория p0); столбцы — мелко проинтегрированные клоны
            offs = s.off + s.sn; rows = {0: s.p0 + offs[:, None] * s.n}; tt = {0: np.zeros(s.m)}; ok = {}
            for sg, nn in ((1, s.nf), (-1, s.nb)):
                if nn:
                    ob_ = s.off + np.linspace(s.sn[0], s.sn[-1], (s.m - 1) * BFINE + 1) if BFINE > 1 else offs   # research-19 BFINE: узлы готовой клетки — каждый BFINE-й клон мелкой цепочки (как при росте), а не грубая цепочка из m клонов: «недоезд» крайнего клона не убивает соседние узлы каскадом
                    if NFDEAD: r_, _, o_ = nf_rows(s.p0, s.u, ob_, nn, sg, strict='mask', brk=False); ok.update({k_: v_[::BFINE] for k_, v_ in o_.items()})
                    else: r_ = nf_rows(s.p0, s.u, ob_, nn, sg, strict=False)[0]
                    rows.update({k_: v_[::BFINE] for k_, v_ in r_.items()}); tt.update({k_: v_[::BFINE] for k_, v_ in nf_rows.TT.items()})
            ks = sorted(rows); s.G = np.array([rows[k] for k in ks]); s.TT = np.array([tt[k] for k in ks]); s.DEAD = ~np.array([ok.get(k, np.ones(s.m, bool)) for k in ks])   # NFDEAD (research-16): узел-заплатка (столбец не пересёк нормаль — у барьера ушёл на другую сторону) — V = BIG навсегда, его четырёхугольники не интерполируются; assert ks == list(range(-s.nb, s.nf + 1)), (ks[0], ks[-1], s.nb, s.nf)   # research-16: TT — время узла вдоль своего столбца (узлы (i,j), (i+1,j) — на одной траектории u)
            s.bb = (s.G[..., 0].min() - 1e-6, s.G[..., 0].max() + 1e-6, s.G[..., 1].min() - 1e-6, s.G[..., 1].max() + 1e-6)
            s.Q = [s.G[:-1, :-1].reshape(-1, 2), s.G[:-1, 1:].reshape(-1, 2), s.G[1:, 1:].reshape(-1, 2), s.G[1:, :-1].reshape(-1, 2)]; return
        for _ in range(s.nf): fw.append(step(fw[-1], s.u))
        y = seg
        for _ in range(s.nb): y = step(y, s.u, -1.); bw.append(y)
        s.G = np.array(bw[::-1] + fw)
        if s.kt > 1: s.G = s.G[sorted(set(range(0, len(s.G), s.kt)) | {len(s.G) - 1})]               # STEPS: строки сетки через kt шагов DTN (концы клетки сохраняются)
        s.bb = (s.G[..., 0].min() - 1e-6, s.G[..., 0].max() + 1e-6, s.G[..., 1].min() - 1e-6, s.G[..., 1].max() + 1e-6)
        s.Q = [s.G[:-1, :-1].reshape(-1, 2), s.G[:-1, 1:].reshape(-1, 2), s.G[1:, 1:].reshape(-1, 2), s.G[1:, :-1].reshape(-1, 2)]   # A0, A1, B1, B0
    def locate(s, Y):
        """для точек Y (n,2): found, it, j, a (поперёк), b (вдоль времени) — четырёхугольник сетки и место в нём."""
        n = len(Y); found = np.zeros(n, bool); qi = np.zeros(n, int); A0, A1, B1, B0 = s.Q
        for a0 in range(0, n, 3000):
            y = Y[a0:a0 + 3000, None, :]; pos = np.ones((y.shape[0], len(A0)), bool); neg = pos.copy()
            for p, q in ((A0, A1), (A1, B1), (B1, B0), (B0, A0)):
                cr = (q[:, 0] - p[:, 0]) * (y[..., 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (y[..., 0] - p[:, 0]); pos &= cr >= -1e-10; neg &= cr <= 1e-10
            m = pos | neg; found[a0:a0 + 3000] = m.any(1); qi[a0:a0 + 3000] = m.argmax(1)
        a = np.full(n, .5); b = np.full(n, .5); p0, p1, p2, p3 = A0[qi], A1[qi], B0[qi], B1[qi]                     # p = (1−a)(1−b)p0 + a(1−b)p1 + (1−a)b p2 + ab p3
        for _ in range(5):
            F = ((1 - a) * (1 - b))[:, None] * p0 + (a * (1 - b))[:, None] * p1 + ((1 - a) * b)[:, None] * p2 + (a * b)[:, None] * p3 - Y
            Fa = (1 - b)[:, None] * (p1 - p0) + b[:, None] * (p3 - p2); Fb = (1 - a)[:, None] * (p2 - p0) + a[:, None] * (p3 - p1); det = Fa[:, 0] * Fb[:, 1] - Fa[:, 1] * Fb[:, 0]; det = np.where(np.abs(det) < 1e-14, 1e-14, det)
            a = np.clip(a - (F[:, 0] * Fb[:, 1] - F[:, 1] * Fb[:, 0]) / det, 0, 1); b = np.clip(b - (Fa[:, 0] * F[:, 1] - Fa[:, 1] * F[:, 0]) / det, 0, 1)
        return found, qi // (s.m - 1), qi % (s.m - 1), a, b
    def incore(s, Y):
        found, it, j, a, b = s.locate(Y); sv = s.sn[j] + a * (s.sn[1] - s.sn[0]); return found & (np.abs(sv) <= s.r + 1e-9)
def shape_ok(c, sl, u, L):
    """ADAPT=1: форма среза по 5 клонам — дискретный гессиан потока. Изгиб среза, растяжение, перекос к потоку, поворот траектории, длина пути."""
    ch = sl[-1] - sl[0]; lc = np.linalg.norm(ch); fc = f(sl[M // 2], u); f0 = f(c.c, u); nf_, n0 = np.linalg.norm(fc), np.linalg.norm(f0)
    if lc < 1e-9 or nf_ < 1e-9: return False
    turn = np.arccos(np.clip(np.dot(fc, f0) / (nf_ * n0), -1, 1)); bend = np.max(np.abs(ch[0] * (sl[1:-1, 1] - sl[0, 1]) - ch[1] * (sl[1:-1, 0] - sl[0, 0]))) / lc ** 2
    trv = abs(ch[0] * fc[1] - ch[1] * fc[0]) / (lc * nf_); stretch = lc / (2 * c.r)
    return turn <= TURN and 1 / STR <= stretch <= STR and bend <= BEND and trv >= TRV and L <= LMAX
def jac(x, u): return np.stack([(f(x + EPSJ * e, u) - f(x - EPSJ * e, u)) / (2 * EPSJ) for e in np.eye(2)], 1)
BBT = np.array([[0., 0.], [0., 1.]])                                                           # управление входит во вторую координату (маятник, ДИ)
def wstep(W, A, h, sg):
    """шаг грамиана на h (A постоянна на шаге; назад по времени A → −A): W ← Φ W Φᵀ + Q, Q = ∫₀ʰ e^{As} BBᵀ e^{Aᵀs} ds до h³ — положительно определена (Эйлер давал det < 0 у старта)."""
    A = sg * A; Ph = np.eye(2) + h * A + h * h / 2 * A @ A; Bq = UB ** 2 * BBT
    return Ph @ W @ Ph.T + h * Bq + h * h / 2 * (A @ Bq + Bq @ A.T) + h ** 3 / 3 * A @ Bq @ A.T
def nerr(err, W, t):
    """ADAPT=2: отклонение среза от линейной модели в единицах «что управление успевает исправить за время t».
    gram — эллипс достижимости W(t) (линеаризация вдоль траектории спо́ры); axis — масштабы UB·t²/2 и UB·t без дрейфа; none — евклидово."""
    if NORM == 'gram': Wi = np.linalg.inv(W + 1e-12 * np.eye(2)); return float(np.sqrt(np.max(np.einsum('ij,jk,ik->i', err, Wi, err))))
    if NORM == 'axis': return float(np.max(np.linalg.norm(err / np.array([UB * t * t / 2, UB * t]), axis=1)))
    return float(np.max(np.linalg.norm(err, axis=1)))
def inbb(c, Y): return (Y[:, 0] >= c.bb[0]) & (Y[:, 0] <= c.bb[1]) & (Y[:, 1] >= c.bb[2]) & (Y[:, 1] <= c.bb[3])
class QIdx:
    """FASTLOC=2 (research-14, по профилю: locate = 58% времени, 674k мелких вызовов): индекс по ЧЕТЫРЁХУГОЛЬНИКАМ сеток клеток.
    Четырёхугольник регистрируется во всех ячейках QB, которые задевает его рамка (с копиями по периоду). Запрос — все точки сразу, векторно."""
    QB = .1
    def __init__(s): s.bins = {}; s.C = []; s.n = 0; s.parts = []; s.dirty = True
    def add(s, c, cid):
        g = c.G; nt, m = g.shape[0], g.shape[1]; A0, A1, B1, B0 = g[:-1, :-1].reshape(-1, 2), g[:-1, 1:].reshape(-1, 2), g[1:, 1:].reshape(-1, 2), g[1:, :-1].reshape(-1, 2)
        it = np.repeat(np.arange(nt - 1), m - 1); j = np.tile(np.arange(m - 1), nt - 1); q0 = s.n; nq = len(A0); s.n += nq
        s.parts.append((A0, A1, B1, B0, np.full(nq, cid), it, j, c.sn[j], np.full(nq, c.sn[1] - c.sn[0]), np.full(nq, c.r), np.full(nq, c.o if hasattr(c, 'o') else 0), np.full(nq, m))); s.dirty = True
        X = np.stack([A0, A1, B1, B0]); lo = X.min(0) - 1e-9; hi = X.max(0) + 1e-9
        for sh in SH:
            if PER is not None and (hi[:, 0].max() + sh < -XL - .2 or lo[:, 0].min() + sh > XL + .2): continue
            ix0 = np.floor((lo[:, 0] + sh) / s.QB).astype(int); ix1 = np.floor((hi[:, 0] + sh) / s.QB).astype(int); iw0 = np.floor(lo[:, 1] / s.QB).astype(int); iw1 = np.floor(hi[:, 1] / s.QB).astype(int)
            for q in range(nq):
                for a_ in range(ix0[q], ix1[q] + 1):
                    for b_ in range(iw0[q], iw1[q] + 1): s.bins.setdefault((a_, b_), []).append((q0 + q, sh))
    def arrays(s):
        if s.dirty and s.parts: s.T = [np.concatenate(z) for z in zip(*s.parts)]; s.dirty = False
        return s.T
    def query(s, Y):
        """все пары (точка, четырёхугольник), где точка внутри: pi, qi, a, b (координаты в четырёхугольнике), sh."""
        if not s.parts: return [np.zeros(0, int)] * 2 + [np.zeros(0)] * 3
        A0, A1, B1, B0 = s.arrays()[:4]; key = np.floor(Y / s.QB).astype(int); kk = key[:, 0] * 100003 + key[:, 1]; order = np.argsort(kk, kind='stable'); uk, st = np.unique(kk[order], return_index=True); st = np.r_[st, len(order)]
        out = []; PI, QI, SH_ = [], [], []; cnt = 0
        for u_, a_, b_ in zip(uk, st[:-1], st[1:]):
            lst = s.bins.get((int(np.floor(Y[order[a_], 0] / s.QB)), int(np.floor(Y[order[a_], 1] / s.QB))))
            if not lst: continue
            qa = np.array(lst); pts = order[a_:b_]; PI.append(np.repeat(pts, len(qa))); QI.append(np.tile(qa[:, 0].astype(int), len(pts))); SH_.append(np.tile(qa[:, 1].astype(float), len(pts))); cnt += len(pts) * len(qa)
            if cnt > 2_000_000: out.append(s._test(Y, np.concatenate(PI), np.concatenate(QI), np.concatenate(SH_))); PI, QI, SH_ = [], [], []; cnt = 0
        if PI: out.append(s._test(Y, np.concatenate(PI), np.concatenate(QI), np.concatenate(SH_)))
        if not out: return [np.zeros(0, int)] * 2 + [np.zeros(0)] * 3
        return [np.concatenate(z) for z in zip(*out)]
    def _test(s, Y, pi, qi, sh):
        A0, A1, B1, B0 = s.arrays()[:4]; y = Y[pi] - np.c_[sh, 0 * sh]; pos = np.ones(len(pi), bool); neg = pos.copy()
        for p_, q_ in ((A0, A1), (A1, B1), (B1, B0), (B0, A0)):
            P_, Q_ = p_[qi], q_[qi]; cr = (Q_[:, 0] - P_[:, 0]) * (y[:, 1] - P_[:, 1]) - (Q_[:, 1] - P_[:, 1]) * (y[:, 0] - P_[:, 0]); pos &= cr >= -1e-10; neg &= cr <= 1e-10
        k = pos | neg; pi, qi, sh, y = pi[k], qi[k], sh[k], y[k]; p0, p1, p2, p3 = A0[qi], A1[qi], B0[qi], B1[qi]; a = np.full(len(pi), .5); b = a.copy()
        for _ in range(5):
            F = ((1 - a) * (1 - b))[:, None] * p0 + (a * (1 - b))[:, None] * p1 + ((1 - a) * b)[:, None] * p2 + (a * b)[:, None] * p3 - y
            Fa = (1 - b)[:, None] * (p1 - p0) + b[:, None] * (p3 - p2); Fb = (1 - a)[:, None] * (p2 - p0) + a[:, None] * (p3 - p1); det = Fa[:, 0] * Fb[:, 1] - Fa[:, 1] * Fb[:, 0]; det = np.where(np.abs(det) < 1e-14, 1e-14, det)
            a = np.clip(a - (F[:, 0] * Fb[:, 1] - F[:, 1] * Fb[:, 0]) / det, 0, 1); b = np.clip(b - (Fa[:, 0] * F[:, 1] - Fa[:, 1] * F[:, 0]) / det, 0, 1)
        return pi, qi, a, b, sh
class Index:
    """сетка ячеек HB: в ячейке — (клетка, сдвиг по периоду); покрытие точки ядрами проверяется только по ним."""
    HB = .25
    def __init__(s): s.bins = {}; s.q = QIdx() if FASTLOC == 2 else None; s.nc = 0
    def add(s, c):
        if s.q is not None: s.q.add(c, s.nc); s.nc += 1; return
        for sh in SH:
            if PER is not None and (c.bb[1] + sh < -XL - .3 or c.bb[0] + sh > XL + .3): continue
            for i in range(int(np.floor((c.bb[0] + sh) / s.HB)), int(np.floor((c.bb[1] + sh) / s.HB)) + 1):
                for j in range(int(np.floor(c.bb[2] / s.HB)), int(np.floor(c.bb[3] / s.HB)) + 1): s.bins.setdefault((i, j), []).append((c, sh))
    def covered(s, Y):
        Y = np.atleast_2d(Y).astype(float).copy(); Y[:, 0] = wrap(Y[:, 0]); m = np.zeros(len(Y), bool)
        if s.q is not None:
            pi, qi, a, b, sh = s.q.query(Y)
            if len(pi): s0, ds, r = s.q.arrays()[7:10]; core = np.abs(s0[qi] + a * ds[qi]) <= r[qi] + 1e-9; m[pi[core]] = True
            return m
        key = np.floor(Y / s.HB).astype(int)
        for k in set(map(tuple, key)):
            q = np.flatnonzero((key[:, 0] == k[0]) & (key[:, 1] == k[1]))
            for c, sh in s.bins.get(k, ()):
                if m[q].all(): break
                yy = Y[q] - np.array([sh, 0.]); g = inbb(c, yy)
                if g.any(): m[q[g]] |= c.incore(yy[g])
        return m
BARRIER = None; BEPS = float(os.environ.get('BEPS', .01)); BTOUCH = int(os.environ.get('BTOUCH', 1))   # BTOUCH=1: отрезок до барьера + BEPS (касание), рост стоп — только внутренние узлы среза у барьера   # CUT (research-14): точки разрыва V из прохода 0 (KD-дерево) — отрезок и рост клетки на них останавливаются
def nearb(P): return np.zeros(len(P), bool) if BARRIER is None else BARRIER.query(np.c_[wrap(P[:, 0]), P[:, 1]], distance_upper_bound=BEPS)[0] < BEPS
OWN = int(os.environ.get('OWN', 0))   # OWN=1 (research-15, по картинке пользователя: щель заполняли цепочкой коротких спор): торец стоп по покрытию СВОИХ столбцов — свободных в строке споры
GROW = int(os.environ.get('GROW', 2)); KF = 41; OVH = float(os.environ.get('OVH', .5)); FRAC = float(os.environ.get('FRAC', .5)); DEPTH = int(os.environ.get('DEPTH', 0)); DFRAC = float(os.environ.get('DFRAC', .5))   # DEPTH=2 (research-15): стоп, когда глубже OVH·h зашла доля края > DFRAC (DEPTH=1 — хоть одна точка, режет клетки рано)   # DEPTH=1 (слово пользователя 2026-10-06): стоп, когда зашли в соседа глубже OVH·h (h — местный шаг узлов)   # GROW=2 (research-14, слово пользователя): рост во все 4 стороны, тормоз по наложению
NORMFRONT = int(os.environ.get('NORMFRONT', 0)); NFEDGE = int(os.environ.get('NFEDGE', 1)); NFLAG = int(os.environ.get('NFLAG', 12)); NFDEAD = int(os.environ.get('NFDEAD', 1)); LOOK = int(os.environ.get('LOOK', 0)); SEEDEPS = float(os.environ.get('SEEDEPS', .005)); NFSUB = int(os.environ.get('NFSUB', 8)); NFMARG = float(os.environ.get('NFMARG', .5))
# NORMFRONT=1 (18б, гипотеза пользователя): строка i клетки — не образ отрезка за i·DTN, а пересечения траекторий-столбцов с прямой через центр c_i ⟂ f(c_i) (торец по нормали к потоку)
import collections; GSTAT = collections.Counter()
NFSTAT = dict(trunc=0, fill=0, nocross=0, mono=0, conv=0)
def quad_convex(ra, rb):
    """все четырёхугольники между строками ra, rb (по столбцам j, j+1) выпуклы (знаки векторных произведений обходов совпадают)."""
    A0, A1, B1, B0 = ra[:-1], ra[1:], rb[1:], rb[:-1]; cs = []
    for p, q, r in ((A0, A1, B1), (A1, B1, B0), (B1, B0, A0), (B0, A0, A1)): cs.append((q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0]))
    cs = np.array(cs); return bool((cs >= -1e-12).all() or (cs <= 1e-12).all())
def nf_rows(p0, u, offs, nmax, sg, strict=True, stop=None, brk=True):
    """строки i = sg·1..nmax: столбцы offs (смещения вдоль n(p0)) интегрируются мелко (DTN/NFSUB, запас по времени ±NFMARG); строка i = пересечения столбцов с прямой через
    c_i (траектория p0 шагом DTN) ⟂ f(c_i). strict: столбец не пересёк, время не растёт или четырёхугольник невыпуклый — стоп (строки до этого); иначе — запасной образ за i·DTN.
    Возвращает (rows {i: (len(offs),2)}, C {i: c_i})."""
    n = normal(p0, u); X0 = np.vstack([p0 + np.asarray(offs)[:, None] * n, p0[None]]); nc = len(offs); K = (int(np.ceil(nmax * (1 + NFMARG))) + NFLAG) * NFSUB + 1
    X = np.empty((K, nc + 1, 2)); X[0] = X0; hs = sg * DTN / NFSUB
    kout = None
    for k in range(1, K):
        X[k] = rk4(X[k - 1], u, hs, 1)
        if kout is None and (not inbox(X[k][nc]) or np.linalg.norm(f(X[k][nc], u)) < FMIN / 4): kout = k                  # центр ушёл — ещё одна строка (как в базе: строка за краем коробки сохраняется, иначе полоса у края не покрыта и очередь сеет в неё без конца)
        if kout is not None and k >= (kout // NFSUB + 1) * NFSUB + NFSUB: X = X[:k + 1]; break
    K = len(X); rows = {}; C = {}; OK = {}; TT = {}; nf_rows.TT = TT; dead = np.zeros(nc, bool); tprev = np.zeros(nc); rprev = X0[:nc]
    for i in range(1, nmax + 1):
        ke = i * NFSUB
        if ke >= K: break
        c = X[ke, nc]; fh = f(c, u); fh = fh / np.linalg.norm(fh)
        if NORMFRONT == 2:                                                                     # research-19 (идея пользователя): торец — ЛОМАНАЯ, почти нормальная каждому клону: от центра c_i наружу, клон j — пересечение его траектории с прямой через узел соседа (ближе к центру) ⟂ потоку В ЭТОМ узле
            ks = np.floor(tprev * NFSUB + 1e-9).astype(int); k0 = int(ks.min()); k1 = min(K - 1, int(ks.max()) + NFLAG * NFSUB); ar = np.arange(k0, k1)
            P = X[ke, :nc].copy(); t = np.full(nc, float(ke)); has = np.zeros(nc, bool); oo = np.asarray(offs); od = np.argsort(oo)
            for side in ([j_ for j_ in od if oo[j_] >= 0], [j_ for j_ in od[::-1] if oo[j_] < 0]):
                pp = c
                for j_ in side:
                    fj = f(pp, u); fj = fj / np.linalg.norm(fj); gj = sg * ((X[k0:k1 + 1, j_] - pp) @ fj); cj = (gj[:-1] <= 0) & (gj[1:] > 0) & (ar >= ks[j_])
                    if not cj.any(): continue
                    kj = int(cj.argmax()); wj = -gj[kj] / (gj[kj + 1] - gj[kj] if gj[kj + 1] != gj[kj] else 1.); has[j_] = True; t[j_] = k0 + kj + wj; P[j_] = X[k0 + kj, j_] * (1 - wj) + X[k0 + kj + 1, j_] * wj; pp = P[j_]
        elif NFLAG:                                                                            # research-16: пересечение строки i — ПЕРВОЕ после пересечения строки i−1 этим же столбцом (у седла столбцы отстают от центра в разы; окно ±NFMARG от времени центра их убивало)
            ks = np.floor(tprev * NFSUB + 1e-9).astype(int); k0 = int(ks.min()); k1 = min(K - 1, int(ks.max()) + NFLAG * NFSUB)
            g = sg * ((X[k0:k1 + 1, :nc] - c) @ fh); cr = (g[:-1] <= 0) & (g[1:] > 0) & (np.arange(k0, k1)[:, None] >= ks[None]); has = cr.any(0); kk = cr.argmax(0); j = np.arange(nc)
        else:
            k0 = max(0, int(ke - NFMARG * ke) - NFSUB); k1 = min(K - 1, int(ke + NFMARG * ke) + NFSUB)
            g = sg * ((X[k0:k1 + 1, :nc] - c) @ fh); cr = (g[:-1] <= 0) & (g[1:] > 0); has = cr.any(0)
            kk = np.where(cr, np.abs(np.arange(k0, k1)[:, None] + .5 - ke), np.inf).argmin(0); j = np.arange(nc)
        if NORMFRONT != 2:
            ga, gb = g[kk, j], g[kk + 1, j]; w = np.where(has, -ga / np.where(gb - ga == 0, 1., gb - ga), 0.); t = k0 + kk + w
            P = X[k0 + kk, j] * (1 - w)[:, None] + X[k0 + kk + 1, j] * w[:, None]
        bad = ~has | (t / NFSUB <= tprev + 1e-9)
        if strict == 'mask':                                                                   # research-16: годность ПО СТОЛБЦАМ (не обрывать всю строку из-за дальнего столбца-кандидата за сепаратрисой)
            dead |= bad; P = np.where(dead[:, None], X[ke, :nc], P); t = np.where(dead, ke, t)
            qb = ~quad_convex_each(rprev, P); dead[:-1] |= qb; dead[1:] |= qb; NFSTAT['fill'] += int(dead.sum())
            if dead[nc // 2] and brk: NFSTAT['trunc'] += 1; break
            rows[sg * i] = P; C[sg * i] = c; OK[sg * i] = ~dead.copy(); TT[sg * i] = sg * t * DTN / NFSUB; tprev = t / NFSUB; rprev = P
            if stop is not None and stop(i, c): break
            continue
        if bad.any():
            if strict: NFSTAT['trunc'] += 1; NFSTAT['nocross'] += int((~has).any()); NFSTAT['mono'] += int((has & (t / NFSUB <= tprev + 1e-9)).any()); break
            NFSTAT['fill'] += int(bad.sum()); P = np.where(bad[:, None], X[ke, :nc], P); t = np.where(bad, ke, t)
        if strict and not quad_convex(rprev, P): NFSTAT['trunc'] += 1; NFSTAT['conv'] += 1; break
        rows[sg * i] = P; C[sg * i] = c; TT[sg * i] = sg * t * DTN / NFSUB; tprev = t / NFSUB; rprev = P
        if stop is not None and stop(i, c): break
    return (rows, C, OK) if strict == 'mask' else (rows, C)
def quad_convex_each(ra, rb):
    """по каждому четырёхугольнику между строками ra, rb (столбцы j, j+1): выпуклый ли."""
    A0, A1, B1, B0 = ra[:-1], ra[1:], rb[1:], rb[:-1]; cs = []
    for p, q, r in ((A0, A1, B1), (A1, B1, B0), (B1, B0, A0), (B0, A0, A1)): cs.append((q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0]))
    cs = np.array(cs); return (cs >= -1e-12).all(0) | (cs <= 1e-12).all(0)
MADAPT = float(os.environ.get('MADAPT', 0)); BFINE = int(os.environ.get('BFINE', 1)); QFAST = int(os.environ.get('QFAST', 1))   # п.30 (research-20): пакетный запрос агента — 3 управления и ветки look одним стеком точек (те же числа, меньше мелких вызовов); 0 — старый путь
SELFOV = int(os.environ.get('SELFOV', 0)); from matplotlib.path import Path as MPath; SMAX = float(os.environ.get('SMAX', 0)); SLAM = float(os.environ.get('SLAM', 0))   # research-19 (идея пользователя): SSIGN = eps > 0 — стоп торца при смене знака d ln w/dt (клетка не проходит минимум ширины у седла)
def grow2(p, u, idx, rm, tm):
    """клетка = прямоугольник индексов [klo,khi]×[ilo,ihi] на мелкой сетке: KF столбцов поперёк (±rm), строки через DTN вперёд/назад ≤ tm.
    Направление (бок ±, торец ±) растёт, пока: в области, изгиб среза (от хорды) в эллипсе достижимости a²·|t|·W ≤ DELTA; упёрлось в соседа
    (край покрыт > FRAC) — добираем наложение OVH·h (h = ширина/(M−1)) или 1 строку и стоп."""
    n = normal(p, u); S = np.linspace(-rm, rm, KF); ds = S[1] - S[0]; k0 = KF // 2; nmax = int(tm / DTN + 1e-9)
    rows = {0: p + S[:, None] * n}; Wt = {0: np.zeros((2, 2))}
    OKM = {}
    if NORMFRONT:
        rows = {0: p + S[:, None] * n}; Wt = {0: np.zeros((2, 2))}
        for sg in (1, -1):
            r_, C_, O_ = nf_rows(p, u, S, nmax, sg, strict='mask'); W = np.zeros((2, 2)); OKM.update(O_)
            for i in range(1, max(abs(k) for k in r_ if k * sg > 0) + 1 if any(k * sg > 0 for k in r_) else 1):
                W = wstep(W, jac(C_[sg * i], u), DTN, sg); rows[sg * i] = r_[sg * i]; Wt[sg * i] = W * (i * DTN)
                if not inbox(C_[sg * i][None]).all() or np.linalg.norm(f(C_[sg * i], u)) < FMIN / 2: break
    else:
        for sg in (1, -1):
            y = rows[0]; W = np.zeros((2, 2))
            for i in range(1, nmax + 1):
                y = step(y, u, float(sg)); A = jac(y[k0], u)
                W = wstep(W, A, DTN, sg)
                rows[sg * i] = y; Wt[sg * i] = W * (i * DTN)                                       # эллипс энергии a²·t (куб |δu| ≤ a внутри него)
                if not inbox(y[k0:k0 + 1]).all() or np.linalg.norm(f(y[k0], u)) < FMIN / 2: break
    imin, imax = min(rows), max(rows); okr = lambda i: OKM.get(i, np.ones(KF, bool))
    def bend(klo, khi, i, Wc):
        if khi - klo < 2: return 0.
        P = rows[i][klo:khi + 1]; ch = P[-1] - P[0]; L = np.linalg.norm(ch)
        if L < 1e-12: return np.inf
        nn = np.array([-ch[1], ch[0]]) / L; d = (P[1:-1] - P[0]) @ nn; v = d[np.argmax(np.abs(d))] * nn
        lam, Q = np.linalg.eigh(Wc); q = Q.T @ v; return float(np.sqrt(np.sum(q * q / np.maximum(lam, 1e-12 * max(lam.max(), 1e-30)))))
    def ok_rect(klo, khi, ilo, ihi):                                                         # один эллипс на всю клетку: за её полную длительность (столько агент в ней едет)
        Wc = Wt[ihi] if ihi >= -ilo else Wt[ilo]; T_ = (ihi - ilo) * DTN; Wc = Wc * (T_ / max(max(ihi, -ilo) * DTN, 1e-9))
        return all(bend(klo, khi, i, Wc) <= DELTA for i in range(ilo, ihi + 1))
    cov0 = idx.covered(rows[0]) | ~inbox(rows[0]) if OWN else None
    def endcov(row, a, b):                                                                    # доля покрытой новой строки торца: при OWN — только по своим столбцам (свободным в строке споры)
        cr = idx.covered(row) | ~inbox(row)
        if OWN:
            o_ = ~cov0[a:b + 1]
            if o_.any(): return cr[o_].mean()
        return cr.mean()
    klo, khi, ilo, ihi = k0 - 1, k0 + 1, 0, 0; act = {'L': True, 'R': True, 'F': True, 'B': True}; extra = {}
    while any(act.values()):
        for d in 'RLFB':
            if not act[d]: continue
            if d in 'RL':
                k = khi + 1 if d == 'R' else klo - 1
                if k < 0 or k >= KF: act[d] = False; continue
                if ihi - ilo < 3 and (act['F'] or act['B']): continue                         # сначала хоть немного длины (эллипс при T → 0 вырожден)
                col = np.array([rows[i][k] for i in range(ilo, ihi + 1)])
                if not all(okr(i)[k] for i in range(ilo, ihi + 1)) or not inbox(col).all() or not ok_rect(min(klo, k), max(khi, k), ilo, ihi) or (BARRIER is not None and nearb(col).any()):
                    GSTAT['side_' + ('mask' if not all(okr(i)[k] for i in range(ilo, ihi + 1)) else 'box' if not inbox(col).all() else 'bend' if not ok_rect(min(klo, k), max(khi, k), ilo, ihi) else 'bar')] += 1; act[d] = False; continue
                if DEPTH:                                                                     # глубина: новый край И точка на OVH·h внутрь (в столбцах — растяжение учтено) покрыты соседом
                    jd = max(1, int(np.ceil(OVH * (khi - klo + 1) / (M - 1)))); kin = k - jd if d == 'R' else k + jd
                    if 0 <= kin < KF:
                        inn = np.array([rows[i][kin] for i in range(ilo, ihi + 1)])
                        dp_ = idx.covered(col) & idx.covered(inn)
                        if (dp_.mean() > DFRAC if DEPTH == 2 else dp_.any()): act[d] = False; continue
                    if d == 'R': khi = k
                    else: klo = k
                    continue
                if d == 'R': khi = k
                else: klo = k
                if d in extra:
                    extra[d] -= 1
                    if extra[d] <= 0: act[d] = False
                elif (idx.covered(col) | ~inbox(col)).mean() > FRAC: extra[d] = int(np.ceil(OVH * (khi - klo) / (M - 1)))
            else:
                i = ihi + 1 if d == 'F' else ilo - 1
                if i not in rows: GSTAT['end_norow'] += 1; act[d] = False; continue
                row = rows[i][klo:khi + 1]
                if not okr(i)[klo:khi + 1].all(): GSTAT['end_mask'] += 1
                elif not ok_rect(klo, khi, min(ilo, i), max(ihi, i)): GSTAT['end_bend'] += 1
                if not okr(i)[klo:khi + 1].all() or not ok_rect(klo, khi, min(ilo, i), max(ihi, i)) or (BARRIER is not None and nearb(row[1:-1]).any()):
                    act[d] = False; continue
                if SELFOV == 2 and self_hit(rows, i, klo, khi, ilo, ihi, rows[0][k0, 0]): GSTAT['end_self'] += 1; act[d] = False; continue   # b4 (п.27): строгий стоп, 0 строк наложения
                if SELFOV == 1 and ihi - ilo >= 4:                                                         # research-19 (слово пользователя): клетка не заезжает на саму себя — новая строка торца не должна попадать в уже выросшую часть ЭТОЙ клетки (без 2 строк у растущего торца; копии через период)
                    lo_, hi_ = (ilo, ihi - 2) if d == 'F' else (ilo + 2, ihi)
                    pg_ = np.array([rows[q][klo] for q in range(lo_, hi_ + 1)] + [rows[q][khi] for q in range(hi_, lo_ - 1, -1)]); pth_ = MPath(pg_)
                    if d not in extra and any(pth_.contains_points(row + np.array([sh_, 0.])).any() for sh_ in SH): GSTAT['end_self'] += 1; extra[d] = 1   # слово пользователя: наезжать на себя можно, но на гало — эта строка добавляется (наложение в 1 строку) и торец останавливается
                if SMAX > 0 or SLAM > 0:                                                              # research-18 (идея пользователя): ширина фронта w(t) — стоп при растяжении/сжатии > SMAX или |d ln w/dt| > SLAM
                    wl = lambda r_: float(np.linalg.norm(np.diff(r_, axis=0), axis=1).sum()) + 1e-12
                    w0_ = wl(rows[0][klo:khi + 1]); wi_ = wl(row); wp_ = wl(rows[ihi if d == 'F' else ilo][klo:khi + 1])
                    if (SMAX > 0 and max(wi_ / w0_, w0_ / wi_) > SMAX) or (SLAM > 0 and abs(np.log(wi_ / wp_)) / DTN > SLAM): GSTAT['end_stretch'] += 1; act[d] = False; continue
                if DEPTH:
                    iin = i - 1 if d == 'F' else i + 1; inn = rows[iin][klo:khi + 1]
                    dp_ = idx.covered(row) & idx.covered(inn)
                    if (dp_.mean() > DFRAC if DEPTH == 2 else dp_.any()): act[d] = False; continue
                    if d == 'F': ihi = i
                    else: ilo = i
                    continue
                if d == 'F': ihi = i
                else: ilo = i
                if d in extra: act[d] = False
                elif endcov(row, klo, khi) > FRAC: extra[d] = 1
    if ihi - ilo == 0 or (S[khi] - S[klo]) / 2 < RMIN: return None
    c = Cell(p + n * (S[klo] + S[khi]) / 2, u, (S[khi] - S[klo]) / 2 / (1 + HALO)); c.n = n; c.nf, c.nb = ihi, -ilo; c.p0 = p; c.off = (S[klo] + S[khi]) / 2
    if MADAPT > 0:                                                                            # research-19 (идея пользователя): число клонов-узлов клетки адаптивно — удваивать (5 → 9 → 17 → 33), пока ломаная торца по m узлам отходит от строки мелкой сетки больше MADAPT (в любой строке клетки)
        Sx = S[klo:khi + 1]
        for m_ in (5, 9, 17, 33):
            sn_ = np.linspace(Sx[0], Sx[-1], m_); dev = 0.
            for i in range(ilo, ihi + 1):
                R_ = rows[i][klo:khi + 1]; dev = max(dev, float(np.hypot(R_[:, 0] - np.interp(Sx, sn_, np.interp(sn_, Sx, R_[:, 0])), R_[:, 1] - np.interp(Sx, sn_, np.interp(sn_, Sx, R_[:, 1]))).max()))
            if dev <= MADAPT: break
        c.m = m_; GSTAT['m_%d' % m_] += 1
    return c
def in_tri(P, A, B, C):
    cr = lambda a, b, p: (b[None, :, 0] - a[None, :, 0]) * (p[:, None, 1] - a[None, :, 1]) - (b[None, :, 1] - a[None, :, 1]) * (p[:, None, 0] - a[None, :, 0])
    d1, d2, d3 = cr(A, B, P), cr(B, C, P), cr(C, A, P)
    return ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
def self_hit(rows, i, klo, khi, ilo, ihi, th0):
    """SELFOV: попадают ли узлы новой строки i (столбцы klo..khi) внутрь четырёхугольников строк q, q+1 этой же клетки, не соседних с i."""
    qs = [q for q in range(ilo, ihi) if (q + 1 <= i - 2 if i > ihi else q >= i + 2)]
    if not qs: return False
    uw = lambda Y: np.stack([th0 + (Y[:, 0] - th0 + PER / 2) % PER - PER / 2, Y[:, 1]], 1) if PER else Y
    R = {j: uw(rows[j][klo:khi + 1]) for j in qs + [qs[-1] + 1]}; P = uw(rows[i][klo:khi + 1])
    A = np.concatenate([R[q][:-1] for q in qs]); B = np.concatenate([R[q][1:] for q in qs]); C = np.concatenate([R[q + 1][1:] for q in qs]); D = np.concatenate([R[q + 1][:-1] for q in qs])
    for k in ((-1, 0, 1) if PER else (0,)):
        Pk = P + np.array([k * (PER or 0.), 0.])
        if (in_tri(Pk, A, B, C) | in_tri(Pk, A, C, D)).any(): return True
    return False
LIMITS = None                                                                                   # REFINE (research-14): функция p → (rmax, tmax) — измельчение по невязке Беллмана
def build_layer(u, rng, log=None):
    cells = []; fails = 0; idx = Index(); queue = []; seeds = []; sc0 = np.linspace(-1, 1, M); bar = tqdm(desc='layer u=%+g cells' % u, unit='cell', mininterval=MI)
    while fails < NFAIL:
        bar.n = len(cells); bar.set_postfix(fails=fails, queue=len(queue), refresh=False); bar.update(0)
        p = queue.pop(0) if queue else np.array([rng.uniform(-XL, XL), rng.uniform(-WL, WL)])
        if not inbox(p): continue
        p = np.array([wrap(p[0]), p[1]])
        if np.linalg.norm(f(p, u)) < FMIN or idx.covered(p[None])[0]: fails += 0 if queue else 1; continue          # у равновесия потока (|f| мало) клетку не строим
        if SEEDEPS > 0 and len(seeds) and np.min(np.abs(np.array(seeds) - p).max(1)) < SEEDEPS: continue   # research-16 SEEDEPS: в точку рядом с прошлым посевом не сеем (цикл почти одинаковых клеток у края: посевы отличались на 1e-6)
        seeds.append(p.copy())
        rm, tm = LIMITS(p, u) if LIMITS else (RMAX, TMAX); n = normal(p, u); ext = []
        if GROW == 2:
            c = grow2(p, u, idx, rm, tm)
            if c is None: fails += 0 if queue else 1; continue
            fails = 0; c.build(); cells.append(c); idx.add(c); g0, g1 = c.G[-1], c.G[0]; yf = g0[len(g0) // 2]; yb = g1[len(g1) // 2]
            for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): yf = step(yf, u); yb = step(yb, u, -1.)
            queue += [yf, yb, c.c + 1.9 * c.r * c.n, c.c - 1.9 * c.r * c.n]
            for e in (g0, g1): e = e[len(e) // 2]; queue += [e + 1.9 * c.r * normal(e, u), e - 1.9 * c.r * normal(e, u)]
            if log: log(u, cells)
            continue
        for sg in (1., -1.):                                                                  # отрезок: до соседа / края / 2·RMAX, с заходом в соседа на OVL
            ss = sg * np.arange(1, int(2 * rm / .01) + 1) * .01; P = p + ss[:, None] * n; bad = ~inbox(P) | idx.covered(P); k = int(np.argmax(bad)) if bad.any() else -1; nb_ = nearb(P); kb = int(np.argmax(nb_)) if nb_.any() else -1
            if kb >= 0 and (k < 0 or kb <= k): ext.append((abs(ss[kb - 1]) if kb > 0 else 0.) + (BEPS if BTOUCH else 0.)); continue      # барьер раньше соседа: до барьера, без захода
            ext.append(((abs(ss[k - 1]) if k > 0 else 0.) + OVL) if k >= 0 else abs(ss[-1]))
        lo, hi = -ext[1], ext[0]
        if hi - lo > 2 * rm: mid = np.clip(0., lo + rm, hi - rm); lo, hi = mid - rm, mid + rm
        if hi - lo < 2 * RMIN: fails += 0 if queue else 1; continue
        for shrink in ((1., .5, .25) if ADAPT and SEL == 'stop' else (1.,)):                 # ADAPT: клетка вышла короткой из-за формы — сужаем отрезок и растим заново
            c = Cell(p + n * (lo + hi) / 2, u, (hi - lo) / 2 * shrink); seg = c.c + (c.r * sc0)[:, None] * c.n; ends = {}; byshape = False; EL = {}; WS = {}; r0 = c.r
            for sg in (1, -1):                                                                # рост по времени: пока в срезе есть непокрытые узлы; первый целиком покрытый срез — курсовое пересечение
                i = 0; sl = seg; L = 0.; ce = c.c + EPSJ * c.n; W = np.zeros((2, 2)); el = [(0., 0.)]; ws = [W]
                while (i + 1) * DTN <= tm + 1e-9:
                    prev = sl[M // 2]; sl = step(sl, u, float(sg)); inb = inbox(sl); L += float(np.linalg.norm(sl[M // 2] - prev))
                    if not inb.any() or np.linalg.norm(f(sl[M // 2], u)) < FMIN / 2: break
                    if ADAPT == 1 and not shape_ok(c, sl, u, L): byshape = True; break
                    if BARRIER is not None and nearb(sl[1:-1] if BTOUCH else sl).any(): break
                    if ADAPT == 2:
                        ce = step(ce, u, float(sg)); A = jac(sl[M // 2], u)
                        W = wstep(W, A, DTN, sg)
                        tng = (ce - sl[M // 2]) / EPSJ; e = nerr(sl - (sl[M // 2] + (c.r * sc0)[:, None] * tng), W, (i + 1) * DTN); el.append((e, L)); ws.append(W)
                        if SEL == 'stop' and e > DELTA: byshape = True; break
                    i += 1
                    if (idx.covered(sl) | ~inb).all(): break
                ends[sg] = i; EL[sg] = np.array(el[:i + 1]); WS[sg] = ws[:i + 1]
            c.nf, c.nb = ends[1], ends[-1]
            if ADAPT == 2 and SEL == 'area':                                                  # ширина из δ: e ∝ r², допустимая полуширина r·√(δ/e); длины вперёд/назад — максимум площади w·(путь)
                ef, eb = EL[1], EL[-1]; wf = np.minimum(c.r, c.r * np.sqrt(DELTA / np.maximum(ef[:, 0], 1e-12))); wb = np.minimum(c.r, c.r * np.sqrt(DELTA / np.maximum(eb[:, 0], 1e-12)))
                Wd = np.minimum(wf[:, None], wb[None, :]); AR = Wd * (ef[:, 1][:, None] + eb[:, 1][None, :]); AR[Wd < RMIN] = -1; AR[0, 0] = -1
                k = np.unravel_index(int(np.argmax(AR)), AR.shape)
                if AR[k] > 0: c.nf, c.nb, c.r = int(k[0]), int(k[1]), float(Wd[k])
                else: c.nf = c.nb = 0
            if not (byshape and c.nb + c.nf < 6) or c.r * .5 < RMIN: break
        if ADAPT == 2 and STEPS and c.nb + c.nf > 0:                                         # m: кусочно-лин. ошибка среза ≈ e/(m−1)² ≤ DN (e ∝ r²); kt: прогиб траектории между строками ≤ DN
            e_end = max(EL[1][c.nf, 0], EL[-1][c.nb, 0]) * (c.r / r0) ** 2; c.m = int(np.clip(1 + np.ceil(np.sqrt(e_end / DN)), 3, MMAX))
            h = (c.nf + c.nb) // 2; Wc = WS[1][min(h, c.nf)] if c.nf >= c.nb else WS[-1][min(h, c.nb)]; c.build(); G0 = c.G; kt = 1
            for k in (8, 6, 4, 2):
                if len(G0) <= k: continue
                dev = (G0[k // 2:len(G0) - k // 2] - (G0[:len(G0) - k] + G0[k:]) / 2).reshape(-1, 2)
                if nerr(dev, Wc, max(h, 1) * DTN) <= DN: kt = k; break
            c.kt = kt if not ((np.abs(wrap(G0[..., 0])) < 3 * RHO) & (np.abs(G0[..., 1]) < 3 * RHO)).any() else 1   # у цели строки не прореживаем (иначе узлы перепрыгивают цель ±RHO)
        if c.nb + c.nf == 0: fails += 0 if queue else 1; continue
        fails = 0; c.build(); cells.append(c); idx.add(c)
        g0, g1 = c.G[-1], c.G[0]; yf = g0[len(g0) // 2]; yb = g1[len(g1) // 2]
        for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2)) if ADAPT else int(.9 * tm / DTN)): yf = step(yf, u); yb = step(yb, u, -1.)
        queue += [yf, yb, c.c + 1.9 * c.r * c.n, c.c - 1.9 * c.r * c.n]
        for e in (g0, g1): e = e[len(e) // 2]; queue += [e + 1.9 * c.r * normal(e, u), e - 1.9 * c.r * normal(e, u)]
        if log: log(u, cells)
    bar.n = len(cells); bar.close(); return cells, idx
class Atlas:
    def __init__(s, seed=0, log=None):
        rng = np.random.default_rng(seed); s.layers = []; s.idx = []
        for u in US: l, ix = build_layer(u, rng, log); s.layers.append(l); s.idx.append(ix)
        s.finish()
    def finish(s):
        s.cells = [c for l in s.layers for c in l]; N = 0
        for c in s.cells: c.o = N; N += c.G.shape[0] * c.m
        s.P = np.concatenate([c.G.reshape(-1, 2) for c in s.cells]); s.N = N; s.goal = ingoal(s.P); s.V = np.full(N, BIG); s.V[s.goal] = 0.
        s.dead = np.concatenate([c.DEAD.ravel() if getattr(c, 'DEAD', None) is not None else np.zeros(c.G.shape[0] * c.m, bool) for c in s.cells]); s.dead &= ~s.goal
    def binidx(s):
        """FASTLOC (research-14): сетка ячеек LB → для каждой клетки и сдвига по периоду — ключи ячеек её рамки; поиск точки — только среди клеток своей ячейки."""
        s.LB = .2; s.NB = int(np.ceil(2 * (WL + 1) / s.LB)) + 2
        for c in s.cells:
            c.bk = {}
            for sh in SH:
                if PER is not None and (c.bb[1] + sh < -XL - 1e-9 or c.bb[0] + sh > XL + 1e-9): continue
                ix = np.arange(int(np.floor((c.bb[0] + sh + XL + 1) / s.LB)), int(np.floor((c.bb[1] + sh + XL + 1) / s.LB)) + 1); iw = np.arange(int(np.floor((c.bb[2] + WL + 1) / s.LB)), int(np.floor((c.bb[3] + WL + 1) / s.LB)) + 1)
                c.bk[sh] = (ix[:, None] * s.NB + iw[None, :]).ravel()
    def stencils(s, Y):
        Y = Y.astype(float).copy(); Y[:, 0] = wrap(Y[:, 0]); I, IDX, W, CR = [], [], [], []
        if FASTLOC == 2:
            if not hasattr(s, 'qx'):
                s.qx = QIdx()
                for ci, c in enumerate(s.cells): s.qx.add(c, ci)
            pi, qi, a, b, sh = s.qx.query(Y)
            if not len(pi): return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4))
            T_ = s.qx.arrays(); it, j, o, m = T_[5][qi], T_[6][qi], T_[10][qi], T_[11][qi]; I = pi
            IDX = o[:, None] + np.stack([it * m + j, it * m + j + 1, (it + 1) * m + j, (it + 1) * m + j + 1], 1); W = np.stack([(1 - b) * (1 - a), (1 - b) * a, b * (1 - a), b * a], 1)
            if CORE: cr = np.abs(T_[7][qi] + a * T_[8][qi]) <= T_[9][qi] + 1e-9; hc = np.zeros(len(Y), bool); hc[I[cr]] = True; k = cr | ~hc[I]; I, IDX, W = I[k], IDX[k], W[k]
            return I, IDX, W
        if FASTLOC:
            if not hasattr(s.cells[0], 'bk'): s.binidx()
            key = np.floor((Y[:, 0] + XL + 1) / s.LB).astype(int) * s.NB + np.floor((Y[:, 1] + WL + 1) / s.LB).astype(int); order = np.argsort(key, kind='stable'); sk = key[order]
        for c in s.cells:
            for sh in (c.bk if FASTLOC else SH):
                if not FASTLOC and PER is not None and (c.bb[1] + sh < -XL or c.bb[0] + sh > XL): continue
                if FASTLOC:
                    kk = c.bk[sh]; lo = np.searchsorted(sk, kk, 'left'); hi = np.searchsorted(sk, kk, 'right'); m_ = hi > lo
                    if not m_.any(): continue
                    q0 = np.concatenate([order[a_:b_] for a_, b_ in zip(lo[m_], hi[m_])]); yy0 = Y[q0] - np.array([sh, 0.]); q = q0[inbb(c, yy0)]
                    if not len(q): continue
                    yy = Y - np.array([sh, 0.])
                else:
                    yy = Y - np.array([sh, 0.]); q = np.flatnonzero(inbb(c, yy))
                if not len(q): continue
                found, it, j, a, b = c.locate(yy[q]); q, it, j, a, b = q[found], it[found], j[found], a[found], b[found]
                if not len(q): continue
                I.append(q); CR.append(np.abs(c.sn[j] + a * (c.sn[1] - c.sn[0])) <= c.r + 1e-9); IDX.append(c.o + np.stack([it * c.m + j, it * c.m + j + 1, (it + 1) * c.m + j, (it + 1) * c.m + j + 1], 1)); W.append(np.stack([(1 - b) * (1 - a), (1 - b) * a, b * (1 - a), b * a], 1))
        if not I: return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4))
        I, IDX, W = np.concatenate(I), np.concatenate(IDX), np.concatenate(W)
        if CORE:                                                                              # CORE=1 (research-14): точка в ядре какой-то клетки — гало/боковой заход не используются (утечка V через разрыв T*)
            cr = np.concatenate(CR); hc = np.zeros(len(Y), bool); hc[I[cr]] = True; k = cr | ~hc[I]; I, IDX, W = I[k], IDX[k], W[k]
        return I, IDX, W
    @staticmethod
    def interp(W, VI, jump=True):
        v = (W * VI).sum(1); m = W > 1e-6; bad = (m & (VI >= BIG / 2)).any(1)
        if JUMP > 0 and jump:
            vm = np.where(m, VI, -np.inf).max(1); vn = np.where(m, VI, np.inf).min(1); j = vm - vn > JUMP
            if JDIR: j = (np.abs(VI[:, 0] - VI[:, 1]) > JUMP) & (np.abs(VI[:, 2] - VI[:, 3]) > JUMP)
            if JMODE == 'drop': bad = bad | j
            else: v = np.where(j, vm if JMODE == 'max' else VI[np.arange(len(VI)), W.argmax(1)], v)
        v[bad] = BIG; return v
    def vstar(s, Y):
        I, IDX, W = s.stencils(Y); out = np.full(len(Y), BIG)
        if len(I): v = s.interp(W, s.V[IDX]); np.minimum.at(out, I, v)
        out[ingoal(Y)] = 0.; return out
    def tgoal(s, Y, u, n=8):
        tg = np.full(len(Y), np.inf); y = Y
        for i in range(1, n + 1): y = rk4(y, u, DTN / n, 1); tg = np.where(np.isinf(tg) & ingoal(y), DTN * i / n, tg)
        return tg
    def solve(s, it=20000):
        E = []; Vg = np.full(s.N, np.inf)
        for u in US: Vg = np.minimum(Vg, s.tgoal(s.P, u)); E.append(s.stencils(step(s.P, u)))
        I = np.concatenate([e[0] for e in E]); IDX = np.concatenate([e[1] for e in E]); W = np.concatenate([e[2] for e in E]); o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]
        st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; V = np.minimum(s.V, Vg); dead = getattr(s, 'dead', np.zeros(s.N, bool)); V[dead] = BIG
        ea, eb, ec = [], [], []                                                               # research-16 NFEDGE: при NORMFRONT шаг DTN узла строки i не попадает в узел строки i+1 (угол строки i ещё BIG → интерполяция отравлена);
        for c in s.cells:                                                                     # узлы (i,j) → (i+1,j) лежат на одной траектории столбца — ребро с точным временем τ(i+1,j) − τ(i,j)
            if getattr(c, 'TT', None) is None or not NFEDGE: continue
            nt, m = c.G.shape[:2]; a = c.o + np.arange((nt - 1) * m); ea.append(a); eb.append(a + m); ec.append(np.maximum(np.diff(c.TT, axis=0).ravel(), 1e-6))
        ea, eb, ec = (np.concatenate(ea), np.concatenate(eb), np.concatenate(ec)) if ea else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
        for n in tqdm(range(it), desc='value iteration', mininterval=MI, leave=False):
            val = DTN + s.interp(W, V[IDX], not JAG); new = V.copy(); new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st))
            if len(ea): np.minimum.at(new, ea, np.where(V[eb] < BIG / 2, ec + V[eb], BIG))
            new[s.goal] = 0.; new[dead] = BIG
            d = np.max(np.abs(new - V)); V = new
            if d < 1e-9: break
        s.V = V; s.n_it = n; s.edges = len(I); return s
    def tgoal3(s, Y, n=8):
        """tgoal по всем 3 управлениям одним стеком → (len(Y), 3)."""
        m = len(Y); y = np.tile(Y, (len(US), 1)); uu = np.repeat(np.array(US), m); tg = np.full(len(y), np.inf)
        for i in range(1, n + 1): y = rk4(y, uu, DTN / n, 1); tg = np.where(np.isinf(tg) & ingoal(y), DTN * i / n, tg)
        return tg.reshape(len(US), m).T
    def step3(s, Y):
        """шаг по всем 3 управлениям одним стеком → (3·len(Y), 2), блоки по управлению."""
        return step(np.tile(Y, (len(US), 1)), np.repeat(np.array(US), len(Y)))
    def look(s, Y, L):
        """LOOK (research-16): цена L шагов жадной политики из Y + V* в конце (rollout-улучшение: барьер на пути виден до того, как агент поверит заниженной V)."""
        Y = Y.copy(); n = len(Y); C = np.zeros(n); done = ingoal(Y); UA = np.array(US)
        if QFAST: return s.look3(Y, L)
        for _ in range(L):
            if done.all(): break
            tgs = np.stack([s.tgoal(Y, u) for u in US], 1); Ys = [step(Y, u) for u in US]; J = np.minimum(tgs, DTN + np.stack([s.vstar(y) for y in Ys], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
            fin = np.isfinite(tg) & ~done; bad = (J.min(1) >= BIG / 2) & ~done
            C += np.where(done, 0., np.where(fin, tg, DTN)); C[bad] = BIG; Y = np.where(done[:, None], Y, np.stack(Ys, 1)[np.arange(n), k]); done |= fin | bad | ingoal(Y)
        return np.minimum(C + np.where(done, 0., s.vstar(Y)), BIG)
    def look3(s, Y, L):
        n = len(Y); C = np.zeros(n); done = ingoal(Y); ar = np.arange(n)
        for _ in range(L):
            if done.all(): break
            tgs = s.tgoal3(Y); Ys = s.step3(Y); J = np.minimum(tgs, DTN + s.vstar(Ys).reshape(len(US), n).T); k = J.argmin(1); tg = tgs[ar, k]
            fin = np.isfinite(tg) & ~done; bad = (J.min(1) >= BIG / 2) & ~done
            C += np.where(done, 0., np.where(fin, tg, DTN)); C[bad] = BIG; Y = np.where(done[:, None], Y, Ys.reshape(len(US), n, 2)[k, ar]); done |= fin | bad | ingoal(Y)
        return np.minimum(C + np.where(done, 0., s.vstar(Y)), BIG)
    def rollout(s, Q, tmax=float(os.environ.get('RTMAX', 30.))):
        Y = np.array(Q, float); n = len(Y); T = np.zeros(n); done = ingoal(Y); sw = np.zeros(n, int); pu = np.full(n, np.nan); path = [Y.copy()]; UA = np.array(US)
        for _ in tqdm(range(int(tmax / DTN)), desc='agent rollout %d starts' % n, mininterval=MI, leave=False):
            if done.all(): break
            if QFAST: tgs = s.tgoal3(Y); Ys3 = s.step3(Y); J = np.minimum(tgs, DTN + (s.look(Ys3, LOOK) if LOOK else s.vstar(Ys3)).reshape(len(US), n).T)
            else: tgs = np.stack([s.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([(s.look(step(Y, u), LOOK) if LOOK else s.vstar(step(Y, u))) for u in US], 1))
            k = J.argmin(1); u = UA[k]; tg = tgs[np.arange(n), k]
            stuck = J.min(1) >= BIG / 2; act = ~done & ~stuck; Yn = np.where(np.isfinite(tg)[:, None], Y, Y) * 0.
            for ui in US:
                m = act & (u == ui)
                if not m.any(): continue
                yy = Y[m]; hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = yy.copy()
                for i in range(1, 9): y_i = rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
                Yn[m] = y8
            sw += act & np.isfinite(pu) & (u != pu); pu = np.where(act, u, pu); Y = np.where(act[:, None], Yn, Y); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= ingoal(Y) | stuck; path.append(Y.copy())
        T[~ingoal(Y)] = np.inf; return T, sw, np.array(path)
def contact_stats(cells, idx):
    ok = []; S = []
    for c in cells:
        g = c.G; nt = len(g); rows = np.unique(np.linspace(0, nt - 1, 7).astype(int)); fr = []; e = .015
        for col, sg in ((g.shape[1] - 1, 1.), (0, -1.)): P = g[rows, col] + sg * e * np.array([normal(p, c.u) for p in g[rows, col]]) * np.sign(np.dot(g[rows[0], -1] - g[rows[0], 0], normal(g[rows[0], col], c.u))); fr.append(float((idx.covered(P) | ~inbox(P)).mean()))
        core = c.c + (c.r * np.linspace(-1, 1, M))[:, None] * c.n; yf, yb = core, core
        for _ in range(c.nf): yf = step(yf, c.u)
        for _ in range(c.nb): yb = step(yb, c.u, -1.)
        for P in (rk4(yf, c.u, DTN / 2, 1), rk4(yb, c.u, -DTN / 2, 1)): fr.append(float((idx.covered(P) | ~inbox(P)).mean()))
        S.append(fr); ok.append(min(fr) >= .8)
    S = np.array(S) if S else np.zeros((0, 4)); return dict(all4=round(float(np.mean(ok)), 3) if ok else None, side=round(float((S[:, :2] >= .8).all(1).mean()), 3) if ok else None, end=round(float((S[:, 2:] >= .8).all(1).mean()), 3) if ok else None)
def starts_ref():
    """старты и эталон: маятник — 100 стартов и pend_ref_T.npy (research-9); ДИ — точное T*."""
    if SYS == 'pend': rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1); rf = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('REFF', 'pend_ref_T.npy'))); rf = rf[:, 0] if rf.ndim == 2 else rf; fin = np.isfinite(rf); return Q[fin], rf[fin]   # стрельба ≤ 3 дуг решает не все старты — нерешённые не сравниваем   # REFF (research-15): эталон для другой цели, напр. pend_shoot_ref_R0.05.npy[:, 0]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from v7_faces_di import tstar_box; Q = np.random.default_rng(1).uniform(-1.5, 1.5, (200, 2)); return Q, tstar_box(Q[:, 0], Q[:, 1], RHO)
if __name__ == '__main__':
    t0 = time.time(); A = Atlas(); tb = time.time() - t0; A.solve(); Q, ref = starts_ref(); ok = ref > .05; T, sw, _ = A.rollout(Q[ok]); fz = np.isfinite(T); r = T[fz] / ref[ok][fz]
    print(json.dumps(dict(SYS=SYS, TMAX=TMAX, OVL=OVL, cells=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), big_nodes=round(float((A.V >= BIG / 2).mean()), 3), reach=round(float(fz.mean()), 3),
                          T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)) if fz.any() else None, sw_med=float(np.median(sw[fz])) if fz.any() else None, sec_build=round(tb, 1), sec=round(time.time() - t0, 1))), flush=True)
