"""ENV4 подводный дрон, база: коридор v6 (обратное адаптивное дерево NB + прямое NF + SLSQP; v6/src/atlas6/adaptive_nd, corridor_nd) — как bench/spores_6d.py для m3d.
Состояние (x, y, θ, vx, vy, ω), 4 слоя-угла тяг {±1}×{±.5}; препятствия (эллипс и круг) — запретом в дереве (blocked) и зазором вдоль пути в SLSQP (clear). Эталон — OCP 20 стартов (v7/reports/env4/data), мерило — ИХ шаг Эйлера .02 (underwaterdrone.py step).
Запуск: python spores_drone.py NB NF K [tag]   → data/drone_spores6d<tag>.json   (aida: ~/spore_v5/a31/spores_2/bench)"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
import numpy as np, multiprocessing as mp
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
NB, NF, K = [int(a) for a in sys.argv[1:4]]; tag = sys.argv[4] if len(sys.argv) > 4 else ''
GM = float(os.environ.get('GMARG', .05)); PAD = float(os.environ.get('PAD', .1)); RHOB = float(os.environ.get('RHOB', .15)); RHOF = float(os.environ.get('RHOF', .15))   # запас на цель (y ≥ 4+GM), запас на препятствия, шаги хеша деревьев
M_, I_, CD, G_, LAT = 1., .1, .05, .5, .2; VM, WM = 3., 3.; XM, YT = 3., 4.; HW = .4; EC, EA, EB = np.array([-3., 2.]), 4., 1.; CC, CR = np.array([1., 2.]), .75
US_ = np.array([(1., .5), (1., -.5), (-1., .5), (-1., -.5)])
def f6(y, u):
    th, vx, vy, w = y[..., 2], y[..., 3], y[..., 4], y[..., 5]; c, s = np.cos(th), np.sin(th); sp = np.sqrt(vx ** 2 + vy ** 2)
    return np.stack([vx, vy, w, (c * u[0] - s * u[1] - CD * sp * vx) / M_, (s * u[0] + c * u[1] - CD * sp * vy - G_) / M_, LAT * u[1] / I_ + 0 * th], -1)
EULER = int(os.environ.get('EULER', 0))   # 1: план на модели среды — явный Эйлер dt .02 (как underwaterdrone.step), вместо RK4
import numba as _nb
@_nb.njit
def _eul1(x, y, th, vx, vy, w, u0, u1, t, CD, M, I, G, LAT):   # b18: тот же явный Эйлер .02 для ОДНОЙ точки (профиль: flow/f6/np.stack = 96–97% запроса)
    sg = 1. if t >= 0 else -1.; n = int(abs(t) / .02 + 1e-9); hr = t - sg * .02 * n
    for k in range(n + 1):
        h = sg * .02 if k < n else hr
        if h == 0.: continue
        c = np.cos(th); s_ = np.sin(th); sp = np.sqrt(vx * vx + vy * vy)
        ax = (c * u0 - s_ * u1 - CD * sp * vx) / M; ay = (s_ * u0 + c * u1 - CD * sp * vy - G) / M; aw = LAT * u1 / I
        vx += ax * h; vy += ay * h; w += aw * h; x += vx * h; y += vy * h; th += w * h
    return x, y, th, vx, vy, w
NUMBA1 = int(os.environ.get('NUMBA1', 0))
def flow(P, s, t, dt_max=.05):
    if EULER and NUMBA1 and np.ndim(P) == 1:
        u = US_[s]; return np.array(_eul1(float(P[0]), float(P[1]), float(P[2]), float(P[3]), float(P[4]), float(P[5]), float(u[0]), float(u[1]), float(t), CD, M_, I_, G_, LAT))
    if EULER:   # сегмент t = n полных шагов .02 + остаток (непрерывно по t — для SLSQP); порядок как в underwaterdrone.step: v += a h; x += v_new h
        P = np.array(P, float); u = US_[s]; sg = 1. if t >= 0 else -1.; n = int(abs(t) / .02 + 1e-9); hs = [sg * .02] * n + [t - sg * .02 * n]
        for h in hs:
            if h == 0: continue
            a = f6(P, u)[..., 3:]; P = P.copy(); P[..., 3:] += a * h; P[..., 0] += P[..., 3] * h; P[..., 1] += P[..., 4] * h; P[..., 2] += P[..., 5] * h
        return P
    return _flow_rk4(P, s, t, dt_max)
def _flow_rk4(P, s, t, dt_max=.05):
    P = np.array(P, float); n = max(1, int(np.ceil(abs(t) / dt_max))); h = t / n; u = US_[s]
    for _ in range(n): k1 = f6(P, u); k2 = f6(P + h / 2 * k1, u); k3 = f6(P + h / 2 * k2, u); k4 = f6(P + h * k3, u); P = P + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return P
def clr(P):   # зазор до препятствий (>0 свободно): эллипс — их формула, круг — расстояние − r
    x, y = P[..., 0], P[..., 1]; dx, dy = x - EC[0], y - EC[1]; rho = np.sqrt((dx / EA) ** 2 + (dy / EB) ** 2) + 1e-12; ct, st = dx / (EA * rho), dy / (EB * rho)
    return np.minimum((rho - 1) * np.sqrt((EA * ct) ** 2 + (EB * st) ** 2), np.hypot(x - CC[0], y - CC[1]) - CR)
wr = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
YG = YT + GM
ing = lambda P: (np.abs(P[..., 0]) <= HW) & (P[..., 1] >= YG)
miss = lambda X: np.maximum(np.abs(X[..., 0]) - HW, 0) + np.maximum(YG - X[..., 1], 0)
g = lambda x: np.array([HW ** 2 - x[0] ** 2, x[1] - YG])
blocked = lambda P: clr(np.asarray(P)) < PAD
okf = lambda p: (abs(p[0]) <= XM) and (-.0 <= p[1] <= YG + .1) and (abs(p[3]) <= VM) and (abs(p[4]) <= VM) and (abs(p[5]) <= WM)
rng = np.random.default_rng(int(os.environ.get('SEED', 0))); seeds = []
while len(seeds) < 32:
    p = np.array([rng.uniform(-HW + .05, HW - .05), rng.uniform(YG, YG + .08), rng.uniform(-np.pi, np.pi), *rng.uniform(-.99 * VM, .99 * VM, 2), rng.uniform(-.99 * WM, .99 * WM)])
    if not blocked(p): seeds.append(p)

# b19 (б): рост деревьев с отбрасыванием рёбер через раздутые препятствия: ECHK=n — n точек вдоль дуги (доли tau), ребро отбрасывается, если любая blocked. ECHK=0 — как было (конец+середина).
import heapq, src.atlas6.adaptive_nd as _and, src.atlas6.corridor_nd as _cnd
ECHK = int(os.environ.get('ECHK', 0)); ECPAD = float(os.environ.get('ECPAD', 0))   # ECPAD: доп. запас на рёбрах (помимо PAD)
def _tree_e(S, tau, starts, N, rho, sw_w, fwd, hist_on=True, hfun=None):
    key = lambda p, s: (s,) + tuple(int(np.floor(p[k] / (rho * S.scale[k]))) for k in range(S.d))
    seen = set(); P = []; G = []; par = []; lay = []; pq = []; cnt = 0; fr = [(j + 1) / ECHK for j in range(ECHK)]
    def add(p, s, g, pa, ls):
        k = key(p, s)
        if k in seen: return -1
        seen.add(k); P.append(p); G.append(g); par.append(pa); lay.append(ls); return len(P) - 1
    def push(i, ns, ls):
        nonlocal cnt
        ch = []; tt = tau if fwd else -tau
        for s in range(S.L):
            e = S.wrap(S.flow(P[i], s, tt))
            if not S.ok(e) or S.blocked(e): continue
            if any(clr(S.flow(P[i], s, tt * f)) < PAD + ECPAD for f in fr[:-1]): continue
            ch.append((s, e))
        hh = hfun(np.array([e for _, e in ch])) if (hfun is not None and ch) else [0.0] * len(ch)
        for (s, e), hv in zip(ch, hh):
            n2 = ns + (1 if (ls >= 0 and s != ls) else 0); cnt += 1
            heapq.heappush(pq, (sw_w * n2 + G[i] + tau + hv, cnt, s, e, G[i] + tau, n2, i))
    for st in starts:
        i = add(S.wrap(st), -1, 0.0, -1, -1); push(i, 0, -1)
    while pq and len(P) < N:
        _, _, s, e, g, n2, pa = heapq.heappop(pq)
        i = add(e, s, g, pa, s)
        if i >= 0: push(i, n2, s)
    return P, G, par, lay
if ECHK: _and._tree = _tree_e; _cnd._tree = _tree_e

# b19: NREF=1 — refine целиком в numba: зазор на КАЖДОМ подшаге Эйлера .02 (число точек на сегмент фиксировано по d0: Lmax_i = int(1.5*d0_i/.02)+2, недостающие — повтор последней), КОНЕЦ — g (окно) ; конечные разности SLSQP по njit-функции.
NREF = int(os.environ.get('NREF', 0)); CSTR = int(os.environ.get('CSTR', 1)); _XP = [0., 0.]   # [pad_x, gm_x] — задаёт _one
CEIL = int(os.environ.get('CEIL', 0))
@_nb.njit
def _cons(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT):
    n_c = 2 + Lm.sum(); out = np.empty(n_c); x, y, th, vx, vy, w = x0[0], x0[1], x0[2], x0[3], x0[4], x0[5]; k = 2
    US0 = np.array([1., 1., -1., -1.]); US1 = np.array([.5, -.5, .5, -.5])
    for j in range(len(seq)):
        u0 = US0[seq[j]]; u1 = US1[seq[j]]; t = d[j]; nst = int(abs(t) / .02 + 1e-9); hr = t - .02 * nst; cnt = 0; lastc = 0.
        for q in range(nst + 1):
            h = .02 if q < nst else hr
            if h == 0.: continue
            c = np.cos(th); s_ = np.sin(th); sp = np.sqrt(vx * vx + vy * vy)
            ax = (c * u0 - s_ * u1 - CD * sp * vx) / M; ay = (s_ * u0 + c * u1 - CD * sp * vy - G) / M; aw = LAT * u1 / I
            vx += ax * h; vy += ay * h; w += aw * h; x += vx * h; y += vy * h; th += w * h
            dx = x - ECx; dy = y - ECy; rho = np.sqrt((dx / EA) ** 2 + (dy / EB) ** 2) + 1e-12; ct = dx / (EA * rho); st = dy / (EB * rho)
            cl = min((rho - 1) * np.sqrt((EA * ct) ** 2 + (EB * st) ** 2), np.sqrt((x - CCx) ** 2 + (y - CCy) ** 2) - CR) - PADT
            if CEIL: cl = min(cl, max(4. - y, HW - abs(x)))   # b19: пересечение потолка y=4 вне окна запрещено (среда зануляет vy у потолка)
            if cnt < Lm[j]: out[k] = cl; k += 1; cnt += 1
            lastc = cl
        while cnt < Lm[j]: out[k] = lastc; k += 1; cnt += 1
    out[0] = HW * HW - x * x; out[1] = y - YGT
    return out
@_nb.njit
def _ksc(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, rho):   # g(2) + KS-зазор на сегмент
    c = _cons(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT); out = np.empty(2 + len(Lm)); out[0] = c[0]; out[1] = c[1]; k = 2
    for j in range(len(Lm)):
        m = c[k]
        for q in range(Lm[j]): m = min(m, c[k + q])
        sm = 0.
        for q in range(Lm[j]): sm += np.exp(-rho * (c[k + q] - m))
        out[2 + j] = m - np.log(sm) / rho; k += Lm[j]
    return out
@_nb.njit
def _ksjac(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, rho):
    b = _ksc(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, rho); J = np.empty((len(b), len(d)))
    for j in range(len(d)):
        dp = d.copy(); dp[j] += 1e-6; J[:, j] = (_ksc(x0, seq, dp, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, rho) - b) / 1e-6
    return J
KSR = float(os.environ.get('KSRHO', 0))
def ksf(c, Lm1):   # KS по сегменту: -(1/ρ) log Σ exp(-ρ clr) ≤ min clr; KSRHO=0 — без агрегации
    if KSR <= 0: return np.concatenate([c[:2], c[2:][::CSTR]])
    out = [c[:2]]; k = 2
    for L in Lm1:
        z = c[k:k + L]; k += L; m = z.min(); out.append([m - np.log(np.exp(-KSR * (z - m)).sum()) / KSR])
    return np.concatenate(out)
def refine_nb(flow, x0, path, g, tries=3, seed=0, clear=None, ns=16, wlim=None):
    from scipy.optimize import minimize
    seq, d0 = _cnd.merge(path); sa = np.array(seq, np.int64); x0a = np.array(x0, float)
    UB = 2. * d0 + .2; Lm1 = (UB / .02).astype(np.int64) + 3
    def cf(d, px, gx): return _cons(x0a, sa, np.asarray(d, float), Lm1, CD, M_, I_, G_, LAT, EC[0], EC[1], EA, EB, CC[0], CC[1], CR, PAD + _XP[0] + px, HW - _XP[1] - gx, YG + _XP[1] + gx)
    px = gx = 0.; ret = None
    for it in range(int(os.environ.get('QITER', 3)) + 1):
        rng = np.random.default_rng(seed); best = (np.inf, None)
        if KSR > 0:
            ar = lambda px_, gx_: (x0a, sa)
            A = lambda d: (x0a, sa, np.asarray(d, float), Lm1, CD, M_, I_, G_, LAT, EC[0], EC[1], EA, EB, CC[0], CC[1], CR, PAD + _XP[0] + px, HW - _XP[1] - gx, YG + _XP[1] + gx, KSR)
            con = {'type': 'ineq', 'fun': lambda d: _ksc(*A(d)), 'jac': lambda d: _ksjac(*A(d))}
        else: con = {'type': 'ineq', 'fun': lambda d: ksf(cf(d, px, gx), Lm1)}
        for k in range(tries):
            di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
            r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, u) for u in UB], constraints=[con], method='SLSQP', options=dict(maxiter=80, ftol=1e-9))
            if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
        if best[1] is None: break
        e = np.round(np.cumsum(best[1]) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); c = cf(dq, px, gx)   # нарушение на КВАНТОВАННОМ плане (как его выполнит verify)
        ret = best
        if c.min() >= -1e-6: break
        if c[2:].min() < 0: px += -c[2:].min() + .01
        if c[:2].min() < 0: gx += -c[:2].min() + .01
    ok = ret is not None
    return (ret[0] if ok else float(d0.sum())), seq, (ret[1] if ok else d0), ok
if NREF: _cnd.refine = refine_nb
EPS = float(os.environ.get('EPS', 0))
def cq_early(S_, flow_, x0, back_, tau, NF_, rho, g_, miss_, K=5, kn=8, neigh=2, clear=None, tries=3, hfun=None, fs=0, wlim=None, nbest=1):
    C, nh = _cnd.candidates(S_, x0, back_, tau, NF_, rho, miss_, kn, hfun=hfun); best = None
    for part in (C[:nh], C[nh:]):
        seen = set(); cnt = 0; lst = [c for c in part if c[2]]
        for idx, (_, sc, p) in enumerate(lst):
            if best is not None and best[0] <= (1 + EPS) * sc: break       # лучший уже не хуже оценки текущего (оценки по возрастанию)
            sq = tuple(_cnd.merge(p)[0])
            if sq in seen: continue
            seen.add(sq); T, sqr, d, ok = _cnd.refine(flow_, x0, p, g_, tries=tries, clear=clear, wlim=wlim)
            if ok and (best is None or T < best[0]): best = (T, sqr, d)
            if len(seen) >= K: break
        if best is not None: break      # промахи — только если попаданий нет
    if best is None: return None
    cur = best
    for _ in range(neigh):
        sq0, d0 = list(cur[1]), list(cur[2])
        nb = [([a] + sq0, [0.3] + d0) for a in range(S_.L) if a != sq0[0]] + [(sq0 + [a], d0 + [0.3]) for a in range(S_.L) if a != sq0[-1]]
        nb += [(sq0[1:], d0[1:])] if len(sq0) > 1 else []
        for sqn, dn in nb:
            T, sqr, d, ok = _cnd.refine(flow_, x0, list(zip(sqn, dn)), g_, tries=1, clear=clear, wlim=wlim)
            if ok and T < cur[0]: cur = (T, sqr, d)
        if cur[0] < best[0]: best = cur
    return best
if EPS > 0: corridor_query = cq_early
S = SysN(flow, 4, (3., 2., np.pi, VM, VM, WM), ing, seeds, per=(0, 0, 2 * np.pi, 0, 0, 0), ok=okf, blocked=blocked)
d = json.load(open(os.path.join(HERE, '..', 'v7', 'reports', 'env4', 'data', 'env4_ref_all.json'))); ks = sorted(d, key=int); X0 = np.array([d[k]['x0'] for k in ks]); X0[:, 2] = wr(X0[:, 2]); Tref = np.array([d[k]['T'] for k in ks])
VTOL = float(os.environ.get('VTOL', 0))
def verify(x0, seq, dts, dt=.02, tmax=40.):
    """ИХ шаг (underwaterdrone.step): Эйлер dt .02, кусочно-постоянное u по времени плана, касание = зазор центра < 0, успех = центр в лунке (y ≥ 4, |x| ≤ .4)."""
    x, y, th, vx, vy, w = [float(v) for v in x0]; ends = np.cumsum(dts); T = 0.; k = 0; tch = False
    while T < tmax:
        while k < len(seq) - 1 and T >= ends[k] - VTOL: k += 1
        if T >= ends[-1] + 3.: return None, 'plan_end', tch
        u = US_[seq[k]] if T < ends[-1] else US_[seq[-1]]; c, s = np.cos(th), np.sin(th); sp = np.hypot(vx, vy)
        ax = (c * u[0] - s * u[1] - CD * sp * vx) / M_; ay = (s * u[0] + c * u[1] - CD * sp * vy - G_) / M_
        vx += ax * dt; vy += ay * dt; w += LAT * u[1] / I_ * dt; x += vx * dt; y += vy * dt; th += w * dt; x = min(max(x, -XM - .01), XM + .01); y = min(max(y, -.01), YT + .01); T += dt
        if y >= YT and abs(x) <= HW: return T, 'ok', tch
        if y < 0: vy = max(vy, 0.)
        if y > YT: vy = min(vy, 0.)
        if x < -XM: vx = max(vx, 0.)
        if x > XM: vx = min(vx, 0.)
        if clr(np.array([x, y])) < 0: tch = True
    return None, 'timeout', tch
t0 = time.perf_counter(); back = build_back(S, .4, NB, RHOB); t_build = time.perf_counter() - t0; Gb = np.array(back[1]); print('build_back NB %d: %.1f с; T_max дерева %.2f с, p90 %.2f, мед %.2f' % (NB, t_build, Gb.max(), np.percentile(Gb, 90), np.median(Gb)), flush=True)
RETRY = int(os.environ.get('RETRY', 0)); EXTRA = float(os.environ.get('XPAD', .05)); XG = float(os.environ.get('XGM', .05))
def _one(i, pad_x, gm_x):
    _XP[0] = pad_x; _XP[1] = gm_x
    gx = lambda x: np.array([(HW - gm_x) ** 2 - x[0] ** 2, x[1] - YG - gm_x]); mx = lambda X: np.maximum(np.abs(X[..., 0]) - (HW - gm_x), 0) + np.maximum(YG + gm_x - X[..., 1], 0)
    t1 = time.perf_counter(); b = corridor_query(S, flow, X0[i], back, .4, NF, RHOF, gx, mx, K=K, tries=int(os.environ.get('TRIES', 3)), kn=8, neigh=int(os.environ.get('NEIGH', 2)), nbest=1, clear=lambda P: clr(P) - PAD - pad_x, wlim=None); dt_ = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt_, T_verify=None, why='no_path', touch=False)
    seq, dts = list(b[1]), np.array(b[2], float)
    if int(os.environ.get('QUANT', 0)):
        e = np.round(np.cumsum(dts) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); k_ = dq > 1e-9; seq = [q for q, kk in zip(seq, k_) if kk]; dts = dq[k_]
    Tv, why, tch = verify(X0[i], seq, list(dts)); return dict(i=i, T_agent=float(b[0]), t_query=dt_, T_verify=Tv, why=why, touch=bool(tch), nseg=len(b[1]))
def work(i):
    r = _one(i, 0., 0.); n = 0; tt = r['t_query']
    while RETRY and (r['T_verify'] is None or r['touch']) and n < RETRY:
        n += 1; r2 = _one(i, EXTRA * n, XG * n); tt += r2['t_query']
        if r2['T_verify'] is not None and (r['T_verify'] is None or (r['touch'] and not r2['touch'])): r = r2
    FB = int(os.environ.get('FBNF', 0))   # b20: запас на no_path — повтор с большим NF/K (старт 3 доходит только при NF>=6400)
    if FB and r['T_verify'] is None and r['why'] == 'no_path':
        global NF, K; nf0, k0 = NF, K; NF, K = FB, int(os.environ.get('FBK', 10)); r2 = _one(i, 0., 0.); NF, K = nf0, k0; tt += r2['t_query']
        if r2['T_verify'] is not None: r = r2; r['fb'] = 1
    r['t_query'] = tt; r['retries'] = n; return r
if __name__ == '__main__':
    if NREF:   # b20: прогрев numba ДО fork (иначе JIT ~1.5 с в каждом воркере на первом запросе — это и был p90 запроса b19)
        _x = np.array(X0[0], float); _sq = np.array([0, 1], np.int64); _dd = np.array([.3, .3]); _Lm = ((2. * _dd + .2) / .02).astype(np.int64) + 3
        _A = (_x, _sq, _dd, _Lm, CD, M_, I_, G_, LAT, EC[0], EC[1], EA, EB, CC[0], CC[1], CR, PAD, HW, YG)
        _cons(*_A); _ksc(*_A, 300.); _ksjac(*_A, 300.); flow(X0[0], 0, .1)
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 16)), len(X0))) as p: res = p.map(work, range(len(X0)), chunksize=1)
    wall = time.perf_counter() - t0; tq = np.array([r['t_query'] for r in res]); ok = [r for r in res if r['T_verify'] is not None]; rat = np.array([r['T_verify'] / Tref[r['i']] for r in ok])
    s = dict(method='коридор v6 NB%d NF%d K%d' % (NB, NF, K), n=len(res), reached=len(ok), touches=int(sum(r['touch'] for r in res)), ratio_med=float(np.median(rat)) if len(rat) else None, ratio_p90=float(np.percentile(rat, 90)) if len(rat) else None, ratio_max=float(rat.max()) if len(rat) else None,
             t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_wall_parallel=wall, nodes_back=int(len(back[0])), T_tree_max=float(Gb.max()), T_tree_p90=float(np.percentile(Gb, 90)), pad=PAD, gmarg=GM, rhob=RHOB, rhof=RHOF, per_start=res, Tref=Tref.tolist())
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True); json.dump(s, open(os.path.join(HERE, 'data', 'drone_spores6d%s.json' % tag), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
