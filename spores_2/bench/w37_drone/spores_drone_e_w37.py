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

# w37: кэш прямого дерева между проходами одного старта (дерево не зависит от pad_x/gm_x; результат побитово тот же), TCACHE=1
TCACHE = int(os.environ.get('TCACHE', 0)); _TC = {}
if TCACHE:
    _t0 = _cnd._tree
    def _tree_c(S_, tau, starts, N, rho, sw_w, fwd, hist_on=True, hfun=None):
        key = (np.asarray(starts[0], float).tobytes(), N, rho, sw_w, fwd, hfun is None)
        if key not in _TC: _TC[key] = _t0(S_, tau, starts, N, rho, sw_w, fwd, hist_on, hfun)
        return _TC[key]
    _cnd._tree = _tree_c
# b20: numba-refine — теперь в v6/src/atlas6/drone_nb.py (под флагом NREF=1; install до fork)
NREF = int(os.environ.get('NREF', 0)); CEIL = int(os.environ.get('CEIL', 0)); _XP = [0., 0.]; KSR = float(os.environ.get('KSRHO', 300))
if NREF:
    from src.atlas6 import drone_nb
    DP = dict(CD=CD, M=M_, I=I_, G=G_, LAT=LAT, EC=EC, EA=EA, EB=EB, CC=CC, CR=CR, PAD=PAD, HW=HW, YG=YG, ceil=CEIL, rho=KSR, qiter=int(os.environ.get('QITER', 5)), xp=_XP)
    drone_nb.install(DP)
S = SysN(flow, 4, (3., 2., np.pi, VM, VM, WM), ing, seeds, per=(0, 0, 2 * np.pi, 0, 0, 0), ok=okf, blocked=blocked)
d = json.load(open(os.path.join(HERE, '..', 'v7', 'reports', 'env4', 'data', 'env4_ref_all.json'))); ks = sorted(d, key=int); X0 = np.array([d[k]['x0'] for k in ks]); X0[:, 2] = wr(X0[:, 2]); Tref = np.array([d[k]['T'] for k in ks])
NSTARTS = int(os.environ.get('NSTARTS', 0))   # b21: >0 — N стартов по распределению env4_ref.starts (seed STARTSEED), без эталона OCP (Tref=1: «дошёл» = verify, T/Tref не имеет смысла)
if NSTARTS:
    _r = np.random.default_rng(int(os.environ.get('STARTSEED', 0))); _S = []
    while len(_S) < NSTARTS:
        _x, _y = _r.uniform(-2.75, 2.75), _r.uniform(.2, 3.6)
        if clr(np.array([_x, _y])) < .1: continue
        _S.append(np.array([_x, _y, _r.uniform(0, 2 * np.pi), *_r.normal(0, .2, 3)]))
    X0 = np.array(_S); X0[:, 2] = wr(X0[:, 2]); Tref = np.ones(NSTARTS)
    if os.environ.get('ONLY'): _o = [int(a) for a in os.environ['ONLY'].split(',')]; X0 = X0[_o]; Tref = Tref[_o]; ONLYIDX = _o
VTOL = float(os.environ.get('VTOL', 0))
DG = {}
HARD = int(os.environ.get('HARD', 0))
def verify(x0, seq, dts, dt=.02, tmax=40.):
    """ИХ шаг (underwaterdrone.step): Эйлер dt .02, кусочно-постоянное u по времени плана, касание = зазор центра < 0, успех = центр в лунке (y ≥ 4, |x| ≤ .4)."""
    x, y, th, vx, vy, w = [float(v) for v in x0]; ends = np.cumsum(dts); T = 0.; k = 0; tch = False; DG.clear(); DG.update(mn=9., mnsub=9., first=None)
    while T < tmax:
        while k < len(seq) - 1 and T >= ends[k] - VTOL: k += 1
        if T >= ends[-1] + 3.: return None, 'plan_end', tch
        u = US_[seq[k]] if T < ends[-1] else US_[seq[-1]]; c, s = np.cos(th), np.sin(th); sp = np.hypot(vx, vy)
        ax = (c * u[0] - s * u[1] - CD * sp * vx) / M_; ay = (s * u[0] + c * u[1] - CD * sp * vy - G_) / M_
        xo, yo = x, y; vx += ax * dt; vy += ay * dt; w += LAT * u[1] / I_ * dt; x += vx * dt; y += vy * dt; th += w * dt; x = min(max(x, -XM - .01), XM + .01); y = min(max(y, -.01), YT + .01); T += dt
        if y >= YT and abs(x) <= HW: return T, 'ok', tch
        if y < 0: vy = max(vy, 0.)
        if y > YT: vy = min(vy, 0.)
        if x < -XM: vx = max(vx, 0.)
        if x > XM: vx = min(vx, 0.)
        c0 = float(clr(np.array([x, y]))); DG['mn'] = min(DG['mn'], c0)
        for a_ in (.25, .5, .75): DG['mnsub'] = min(DG['mnsub'], float(clr(np.array([xo + a_ * (x - xo), yo + a_ * (y - yo)]))))
        if c0 < 0 or min(DG['mnsub'], 9) < 0:
            if c0 >= 0 and DG['first'] is None: DG['first'] = dict(T=round(T, 2), seg=k, nseg=len(seq), xy=[round(x, 3), round(y, 3)], kind='между шагами .02')
            if c0 < 0 and (DG['first'] is None or DG['first']['kind'].startswith('между')):
                if DG['first'] is None: DG['first'] = dict(T=round(T, 2), seg=k, nseg=len(seq), xy=[round(x, 3), round(y, 3)], kind='в узле шага')
            tch = tch or (c0 < 0)
        if c0 < 0: DG['depth'] = min(DG.get('depth', 0.), c0)
    return None, 'timeout', tch
t0 = time.perf_counter(); back = build_back(S, .4, NB, RHOB); t_build = time.perf_counter() - t0; Gb = np.array(back[1]); print('build_back NB %d: %.1f с; T_max дерева %.2f с, p90 %.2f, мед %.2f' % (NB, t_build, Gb.max(), np.percentile(Gb, 90), np.median(Gb)), flush=True)
HFW = float(os.environ.get('HFW', 0)); HFD = float(os.environ.get('HFD', 1))   # w37: A*-эвристика прямого дерева «до обратного» = HFW·(G_b ближайшего узла + HFD·расст. в масштабе)
if HFW > 0:
    from scipy.spatial import cKDTree
    _Pb, _Gb, _tr = back[0], np.array(back[1]), back[5]
    def HF(X):
        d, j = _tr.query(np.asarray(X) / S.scale, k=1); return HFW * (_Gb[np.asarray(j) % len(_Gb)] + HFD * d)
else: HF = None
FB = (int(os.environ['FBNF']), int(os.environ.get('FBK', 10))) if os.environ.get('FBNF') else None; RETRY = int(os.environ.get('RETRY', 0)); EXTRA = float(os.environ.get('XPAD', .05)); XG = float(os.environ.get('XGM', .05))
def _one(i, pad_x, gm_x):
    _XP[0] = pad_x; _XP[1] = gm_x
    gx = lambda x: np.array([(HW - gm_x) ** 2 - x[0] ** 2, x[1] - YG - gm_x]); mx = lambda X: np.maximum(np.abs(X[..., 0]) - (HW - gm_x), 0) + np.maximum(YG + gm_x - X[..., 1], 0)
    def _vfy(seq, dts):
        seq = list(seq); dts = np.array(dts, float)
        if int(os.environ.get('QUANT', 0)):
            e = np.round(np.cumsum(dts) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); k_ = dq > 1e-9; seq = [q for q, kk in zip(seq, k_) if kk]; dts = dq[k_]
        Tv, why, tch = verify(X0[i], seq, list(dts)); return Tv is not None and not tch
    t1 = time.perf_counter(); b = corridor_query(S, flow, X0[i], back, .4, NF, RHOF, gx, mx, K=K, hfun=HF, tries=int(os.environ.get('TRIES', 3)), kn=8, neigh=int(os.environ.get('NEIGH', 2)), nbest=1, clear=lambda P: clr(P) - PAD - pad_x, wlim=None, fb=FB, vfy=(_vfy if int(os.environ.get('VFY', 0)) else None), kfb=(int(os.environ['KFB']) if os.environ.get('KFB') else None)); dt_ = time.perf_counter() - t1
    if b is None:
        dgn = None
        if os.environ.get('DIAG'):
            from src.atlas6.corridor_nd import candidates, refine
            C, nh = candidates(S, X0[i], back, .4, NF, RHOF, mx, 8); nr = 0; nok = 0
            for _, _, p in C[:nh][:5]:
                if not p: continue
                T_, sq_, d_, ok_ = refine(flow, X0[i], p, gx, tries=3, clear=lambda P: clr(P) - PAD - pad_x, wlim=None); nr += 1; nok += bool(ok_)
            hs = [float(c[0]) for c in C]
            dgn = dict(nC=len(C), nh=nh, hmin=(min(hs) if hs else None), nref=nr, nok=nok, clr0=float(clr(X0[i][:2])))
        return dict(i=i, T_agent=None, t_query=dt_, T_verify=None, why='no_path', touch=False, dgn=dgn)
    seq, dts = list(b[1]), np.array(b[2], float)
    if int(os.environ.get('QUANT', 0)):
        e = np.round(np.cumsum(dts) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); k_ = dq > 1e-9; seq = [q for q, kk in zip(seq, k_) if kk]; dts = dq[k_]
    Tv, why, tch = verify(X0[i], seq, list(dts)); dg = dict(DG); sub = dg.get('mnsub', 9) < 0
    if HARD and tch: Tv, why = None, 'touch'   # жёсткий отказ: план с касанием = не дошёл
    return dict(i=i, T_agent=float(b[0]), t_query=dt_, T_verify=Tv, why=why, touch=bool(tch), nseg=len(b[1]), diag=dg, pad_x=pad_x)
def work(i):
    _TC.clear()
    r = _one(i, 0., 0.); n = 0; tt = r['t_query']
    while RETRY and (r['T_verify'] is None or r['touch']) and n < RETRY and not (HARD and r['T_verify'] is not None):
        n += 1; r2 = _one(i, EXTRA * n, XG * n); tt += r2['t_query']
        if (r2['T_verify'] is not None and (r['T_verify'] is None or (r['touch'] and not r2['touch']))) or (HARD and r['T_verify'] is None and r2['T_verify'] is not None): r = r2
    r['t_query'] = tt; r['retries'] = n; return r
if __name__ == '__main__':
    t0 = time.perf_counter()
    if int(os.environ.get('PROCS', 16)) <= 1:   # w37: последовательно в главном процессе (нужно для PARQ — вложенный Pool из демонического воркера нельзя); cpu = свой + дочерние
        import resource; work(0); res = []
        for i_ in range(len(X0)):
            ru0 = resource.getrusage(resource.RUSAGE_SELF); rc0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.perf_counter(); r_ = work(i_)
            ru1 = resource.getrusage(resource.RUSAGE_SELF); rc1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            r_['wall'] = time.perf_counter() - w0; r_['cpu'] = (ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime) + (rc1.ru_utime + rc1.ru_stime - rc0.ru_utime - rc0.ru_stime); res.append(r_)
    else:
        with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 16)), len(X0))) as p: res = p.map(work, range(len(X0)), chunksize=1)
    if os.environ.get('ONLY'):
        for r_, i_ in zip(res, ONLYIDX): r_['i_start'] = i_
    wall = time.perf_counter() - t0; tq = np.array([r['t_query'] for r in res]); ok = [r for r in res if r['T_verify'] is not None]; rat = np.array([r['T_verify'] / Tref[r['i']] for r in ok])
    s = dict(method='коридор v6 NB%d NF%d K%d' % (NB, NF, K), n=len(res), reached=len(ok), touches=int(sum(r['touch'] for r in res)), ratio_med=float(np.median(rat)) if len(rat) else None, ratio_p90=float(np.percentile(rat, 90)) if len(rat) else None, ratio_max=float(rat.max()) if len(rat) else None,
             t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_wall_parallel=wall, nodes_back=int(len(back[0])), T_tree_max=float(Gb.max()), T_tree_p90=float(np.percentile(Gb, 90)), pad=PAD, gmarg=GM, rhob=RHOB, rhof=RHOF, per_start=res, Tref=Tref.tolist())
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True); json.dump(s, open(os.path.join(HERE, 'data', 'drone_spores6d%s.json' % tag), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
