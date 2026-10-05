"""research-13 (слово пользователя 2026-10-05): растущие споры-клетки для ЛЮБОЙ 2D системы (обобщение di_grow_cells.py). По одному атласу на управление.
Спора = точка + нормальный отрезок (M узлов), пронесённый потоком своего управления вперёд и назад ≤ TMAX. Посев встык, обязательные боковые (OVL) и курсовые
пересечения с соседями. Формул системы нет: поток — rk4, положение точки в клетке — поиск четырёхугольника сетки узлов + обратная билинейная карта.
Цена: V(узел) = min_u [Δt + V*(φ_u(узел, Δt))], V*(точка) = min по клеткам любого атласа билинейно; агент каждые Δt берёт argmin.
Системы: SYS=pend (φ̈ = sin φ + u, |u| ≤ .3, цель — верх ±RHO, φ периодична), SYS=di (проверка против di_grow_cells.py)."""
import numpy as np, sys, os, json, time
SYS = os.environ.get('SYS', 'pend'); M = 5; BIG = 1e3; HALO = .1
DTN = float(os.environ.get('DTN', .06)); RMAX = float(os.environ.get('RMAX', .1)); TMAX = float(os.environ.get('TMAX', 1.5)); OVL = float(os.environ.get('OVL', .05)); RHO = float(os.environ.get('RHO', .1))
ADAPT = int(os.environ.get('ADAPT', 0)); TURN = np.deg2rad(float(os.environ.get('TURN', 40))); STR = float(os.environ.get('STR', 2.)); BEND = float(os.environ.get('BEND', .05)); TRV = float(os.environ.get('TRV', .5)); LMAX = float(os.environ.get('LMAX', 1.2))
CORE = int(os.environ.get('CORE', 0)); JUMP = float(os.environ.get('JUMP', 0)); JMODE = os.environ.get('JMODE', 'max'); JAG = int(os.environ.get('JAG', 0)); JDIR = int(os.environ.get('JDIR', 0));   # JDIR=1: разрыв — только скачок поперёк столбцов в ОБЕИХ строках (барьер вдоль потока); JMODE drop — четырёхугольник не используется
   # JMODE near — значение узла с наибольшим весом; JAG=1 — только у агента (vstar), не в счёте V
   # JUMP > 0 (research-14): узлы четырёхугольника расходятся по V > JUMP — разрыв цены, берём max узлов (не обещать лишнего)
NORM = os.environ.get('NORM', 'gram'); DELTA = float(os.environ.get('DELTA', .1)); SEL = os.environ.get('SEL', 'stop'); EPSJ = 1e-5; STEPS = int(os.environ.get('STEPS', 0)); DN = float(os.environ.get('DN', .003)); MMAX = int(os.environ.get('MMAX', 9))   # STEPS=1: узлы поперёк m и шаг строк kt — из той же метрики (ошибка интерполяции ≤ DN); ADAPT=2 (research-14): ошибка линейной модели среза в локальной метрике
RMIN = float(os.environ.get('RMIN', .01)); FMIN = float(os.environ.get('FMIN', .2)); NFAIL = int(os.environ.get('NFAIL', 400))
if SYS == 'pend':
    UM = .3; UB = UM; US = (-UM, 0., UM); PER = 2 * np.pi; XL, WL = np.pi, float(os.environ.get('WS', 3.5))
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
    def __init__(s, c, u, r): s.c, s.u, s.r = np.array(c, float), u, r; s.n = normal(s.c, u); s.nb = 0; s.nf = 0; s.m = M; s.kt = 1
    def build(s):
        """сетка узлов G (nt, M, 2): отрезок с гало, пронесённый потоком на nb шагов назад и nf вперёд."""
        s.sn = np.linspace(-(1 + HALO) * s.r, (1 + HALO) * s.r, s.m); seg = s.c + s.sn[:, None] * s.n; fw = [seg]; bw = []
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
def nerr(err, W, t):
    """ADAPT=2: отклонение среза от линейной модели в единицах «что управление успевает исправить за время t».
    gram — эллипс достижимости W(t) (линеаризация вдоль траектории спо́ры); axis — масштабы UB·t²/2 и UB·t без дрейфа; none — евклидово."""
    if NORM == 'gram': Wi = np.linalg.inv(W + 1e-12 * np.eye(2)); return float(np.sqrt(np.max(np.einsum('ij,jk,ik->i', err, Wi, err))))
    if NORM == 'axis': return float(np.max(np.linalg.norm(err / np.array([UB * t * t / 2, UB * t]), axis=1)))
    return float(np.max(np.linalg.norm(err, axis=1)))
def inbb(c, Y): return (Y[:, 0] >= c.bb[0]) & (Y[:, 0] <= c.bb[1]) & (Y[:, 1] >= c.bb[2]) & (Y[:, 1] <= c.bb[3])
class Index:
    """сетка ячеек HB: в ячейке — (клетка, сдвиг по периоду); покрытие точки ядрами проверяется только по ним."""
    HB = .25
    def __init__(s): s.bins = {}
    def add(s, c):
        for sh in SH:
            if PER is not None and (c.bb[1] + sh < -XL - .3 or c.bb[0] + sh > XL + .3): continue
            for i in range(int(np.floor((c.bb[0] + sh) / s.HB)), int(np.floor((c.bb[1] + sh) / s.HB)) + 1):
                for j in range(int(np.floor(c.bb[2] / s.HB)), int(np.floor(c.bb[3] / s.HB)) + 1): s.bins.setdefault((i, j), []).append((c, sh))
    def covered(s, Y):
        Y = np.atleast_2d(Y).astype(float).copy(); Y[:, 0] = wrap(Y[:, 0]); m = np.zeros(len(Y), bool); key = np.floor(Y / s.HB).astype(int)
        for k in set(map(tuple, key)):
            q = np.flatnonzero((key[:, 0] == k[0]) & (key[:, 1] == k[1]))
            for c, sh in s.bins.get(k, ()):
                if m[q].all(): break
                yy = Y[q] - np.array([sh, 0.]); g = inbb(c, yy)
                if g.any(): m[q[g]] |= c.incore(yy[g])
        return m
BARRIER = None; BEPS = float(os.environ.get('BEPS', .02)); BTOUCH = int(os.environ.get('BTOUCH', 0))   # BTOUCH=1: отрезок до барьера + BEPS (касание), рост стоп — только внутренние узлы среза у барьера   # CUT (research-14): точки разрыва V из прохода 0 (KD-дерево) — отрезок и рост клетки на них останавливаются
def nearb(P): return np.zeros(len(P), bool) if BARRIER is None else BARRIER.query(np.c_[wrap(P[:, 0]), P[:, 1]], distance_upper_bound=BEPS)[0] < BEPS
LIMITS = None                                                                                   # REFINE (research-14): функция p → (rmax, tmax) — измельчение по невязке Беллмана
def build_layer(u, rng, log=None):
    cells = []; fails = 0; idx = Index(); queue = []; sc0 = np.linspace(-1, 1, M)
    while fails < NFAIL:
        p = queue.pop(0) if queue else np.array([rng.uniform(-XL, XL), rng.uniform(-WL, WL)])
        if not inbox(p): continue
        p = np.array([wrap(p[0]), p[1]])
        if np.linalg.norm(f(p, u)) < FMIN or idx.covered(p[None])[0]: fails += 0 if queue else 1; continue          # у равновесия потока (|f| мало) клетку не строим
        rm, tm = LIMITS(p, u) if LIMITS else (RMAX, TMAX); n = normal(p, u); ext = []
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
                        for _ in range(2): W = W + DTN / 2 * (sg * (A @ W + W @ A.T) + UB ** 2 * BBT)
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
    return cells, idx
class Atlas:
    def __init__(s, seed=0, log=None):
        rng = np.random.default_rng(seed); s.layers = []; s.idx = []
        for u in US: l, ix = build_layer(u, rng, log); s.layers.append(l); s.idx.append(ix)
        s.finish()
    def finish(s):
        s.cells = [c for l in s.layers for c in l]; N = 0
        for c in s.cells: c.o = N; N += c.G.shape[0] * c.m
        s.P = np.concatenate([c.G.reshape(-1, 2) for c in s.cells]); s.N = N; s.goal = ingoal(s.P); s.V = np.full(N, BIG); s.V[s.goal] = 0.
    def stencils(s, Y):
        Y = Y.astype(float).copy(); Y[:, 0] = wrap(Y[:, 0]); I, IDX, W, CR = [], [], [], []
        for c in s.cells:
            for sh in SH:
                if PER is not None and (c.bb[1] + sh < -XL or c.bb[0] + sh > XL): continue
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
        st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; V = np.minimum(s.V, Vg)
        for n in range(it):
            val = DTN + s.interp(W, V[IDX], not JAG); new = V.copy(); new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st)); new[s.goal] = 0.
            d = np.max(np.abs(new - V)); V = new
            if d < 1e-9: break
        s.V = V; s.n_it = n; s.edges = len(I); return s
    def rollout(s, Q, tmax=30.):
        Y = np.array(Q, float); n = len(Y); T = np.zeros(n); done = ingoal(Y); sw = np.zeros(n, int); pu = np.full(n, np.nan); path = [Y.copy()]; UA = np.array(US)
        for _ in range(int(tmax / DTN)):
            if done.all(): break
            tgs = np.stack([s.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([s.vstar(step(Y, u)) for u in US], 1)); k = J.argmin(1); u = UA[k]; tg = tgs[np.arange(n), k]
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
    if SYS == 'pend': rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1); return Q, np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pend_ref_T.npy'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from v7_faces_di import tstar_box; Q = np.random.default_rng(1).uniform(-1.5, 1.5, (200, 2)); return Q, tstar_box(Q[:, 0], Q[:, 1], RHO)
if __name__ == '__main__':
    t0 = time.time(); A = Atlas(); tb = time.time() - t0; A.solve(); Q, ref = starts_ref(); ok = ref > .05; T, sw, _ = A.rollout(Q[ok]); fz = np.isfinite(T); r = T[fz] / ref[ok][fz]
    print(json.dumps(dict(SYS=SYS, TMAX=TMAX, OVL=OVL, cells=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), big_nodes=round(float((A.V >= BIG / 2).mean()), 3), reach=round(float(fz.mean()), 3),
                          T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)) if fz.any() else None, sw_med=float(np.median(sw[fz])) if fz.any() else None, sec_build=round(tb, 1), sec=round(time.time() - t0, 1))), flush=True)
