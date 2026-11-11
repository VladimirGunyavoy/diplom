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
rng = np.random.default_rng(0); seeds = []
while len(seeds) < 32:
    p = np.array([rng.uniform(-HW + .05, HW - .05), rng.uniform(YG, YG + .08), rng.uniform(-np.pi, np.pi), *rng.uniform(-.99 * VM, .99 * VM, 2), rng.uniform(-.99 * WM, .99 * WM)])
    if not blocked(p): seeds.append(p)
S = SysN(flow, 4, (3., 2., np.pi, VM, VM, WM), ing, seeds, per=(0, 0, 2 * np.pi, 0, 0, 0), ok=okf, blocked=blocked)
d = json.load(open(os.path.join(HERE, '..', 'v7', 'reports', 'env4', 'data', 'env4_ref_all.json'))); ks = sorted(d, key=int); X0 = np.array([d[k]['x0'] for k in ks]); X0[:, 2] = wr(X0[:, 2]); Tref = np.array([d[k]['T'] for k in ks])
def verify(x0, seq, dts, dt=.02, tmax=40.):
    """ИХ шаг (underwaterdrone.step): Эйлер dt .02, кусочно-постоянное u по времени плана, касание = зазор центра < 0, успех = центр в лунке (y ≥ 4, |x| ≤ .4)."""
    x, y, th, vx, vy, w = [float(v) for v in x0]; ends = np.cumsum(dts); T = 0.; k = 0; tch = False
    while T < tmax:
        while k < len(seq) - 1 and T >= ends[k]: k += 1
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
def work(i):
    t1 = time.perf_counter(); b = corridor_query(S, flow, X0[i], back, .4, NF, RHOF, g, miss, K=K, tries=3, kn=8, neigh=2, nbest=1, clear=lambda P: clr(P) - PAD, wlim=None); dt_ = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt_, T_verify=None, why='no_path', touch=False)
    seq, dts = list(b[1]), np.array(b[2], float)
    if int(os.environ.get('QUANT', 0)):   # концы сегментов — на сетку их шага .02 (их u переключается по времени на сетке)
        e = np.round(np.cumsum(dts) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); k_ = dq > 1e-9; seq = [q for q, kk in zip(seq, k_) if kk]; dts = dq[k_]
    Tv, why, tch = verify(X0[i], seq, list(dts)); return dict(i=i, T_agent=float(b[0]), t_query=dt_, T_verify=Tv, why=why, touch=bool(tch), nseg=len(b[1]))
if __name__ == '__main__':
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 16)), len(X0))) as p: res = p.map(work, range(len(X0)), chunksize=1)
    wall = time.perf_counter() - t0; tq = np.array([r['t_query'] for r in res]); ok = [r for r in res if r['T_verify'] is not None]; rat = np.array([r['T_verify'] / Tref[r['i']] for r in ok])
    s = dict(method='коридор v6 NB%d NF%d K%d' % (NB, NF, K), n=len(res), reached=len(ok), touches=int(sum(r['touch'] for r in res)), ratio_med=float(np.median(rat)) if len(rat) else None, ratio_p90=float(np.percentile(rat, 90)) if len(rat) else None, ratio_max=float(rat.max()) if len(rat) else None,
             t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_wall_parallel=wall, nodes_back=int(len(back[0])), T_tree_max=float(Gb.max()), T_tree_p90=float(np.percentile(Gb, 90)), pad=PAD, gmarg=GM, rhob=RHOB, rhof=RHOF, per_start=res, Tref=Tref.tolist())
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True); json.dump(s, open(os.path.join(HERE, 'data', 'drone_spores6d%s.json' % tag), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
