"""research hub-research-3: почему запросы медленные. Профиль одного запроса corridor_query (+replay_value_fast) по этапам.
Запуск из spores_2/v6 (или ~/spore_v6 на aida): python3 ../v5chain/reports/research/profile_query.py SYS [NQ] [NB NF K]
SYS: dd | pend | manip | manip_obst. Вывод — JSON-строка в stdout (этапы, время, вызовы потока точка/батч, SLSQP nfev) + top cProfile в stderr."""
import sys, time, json, os, platform, cProfile, pstats, io; sys.path.insert(0, '.')
import numpy as np, scipy
import src.atlas6.adaptive_nd as A
import src.atlas6.corridor_nd as C
from scipy import optimize

SYS = sys.argv[1]; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T = {}; N = {}                                         # этап → сек, счётчики


def timed(mod, name, key):
    f0 = getattr(mod, name)
    def w(*a, **k):
        t = time.perf_counter(); r = f0(*a, **k); T[key] = T.get(key, 0) + time.perf_counter() - t; N[key] = N.get(key, 0) + 1; return r
    setattr(mod, name, w)


def count_flow(fl):
    def w(P, s, t):
        P = np.asarray(P); b = 'flow_batch' if P.ndim > 1 else 'flow_point'
        t0 = time.perf_counter(); r = fl(P, s, t); T[b] = T.get(b, 0) + time.perf_counter() - t0; N[b] = N.get(b, 0) + 1
        if P.ndim > 1: N['batch_rows'] = N.get('batch_rows', 0) + P.shape[0]
        return r
    return w


_min = optimize.minimize
NIT0 = []; SL = {}                                                # статус SLSQP → [число, сек, nfev]
def minimize_c(*a, **k):
    if os.environ.get('MAXIT'): k['options'] = dict(k.get('options', {}), maxiter=int(os.environ['MAXIT']))
    t = time.perf_counter(); r = _min(*a, **k); NIT0.append(int(r.nit)) if r.status == 0 else None; e = SL.setdefault(int(r.status), [0, 0.0, 0]); e[0] += 1; e[1] += time.perf_counter() - t; e[2] += int(r.nfev); N['slsqp_nit'] = N.get('slsqp_nit', 0) + int(r.nit); N['slsqp_nfev'] = N.get('slsqp_nfev', 0) + int(r.nfev); return r
C.minimize = minimize_c

if SYS == 'dd':
    from src.atlas6.dd3 import flow
    from src.atlas6.dd_atlas import LAYERS
    NB, NF, K = 1200, 50, 5; tau = 0.25; R = .25; Rth = .26; rho = 0.08; clear = None
    fl = lambda P, s, t: flow(P, LAYERS[s], t)
    dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
    ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
    g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
    miss = lambda X: np.maximum(np.hypot(X[..., 0], X[..., 1]) - R, 0) + np.maximum(np.abs(dth(X[..., 2])) - Rth, 0)
    flc = count_flow(fl)
    S = A.SysN(flc, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
    rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(NQ)]
elif SYS == 'pend':
    from src.atlas6.gcell import rk4, pendulum
    u = 0.3; tau = 0.25; Rg = 0.5; rho = 0.05; f = pendulum(1.0); NB, NF, K = 400, 50, 5; clear = None
    dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
    fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t, dt_max=0.05)
    if os.environ.get('SCALAR'):                       # прототип: точка — чистый Python (math), батч — rk4 numpy
        import math
        def fl(P, s, t, _fl=fl):
            P = np.asarray(P, float)
            if P.ndim > 1: return _fl(P, s, t)
            uu = u * (1 if s == 0 else -1); th, w = float(P[0]), float(P[1]); n = max(1, int(math.ceil(abs(t) / 0.05))); h = t / n
            for _ in range(n):
                a1 = -math.sin(th) + uu; a2 = -math.sin(th + h / 2 * w) + uu; a3 = -math.sin(th + h / 2 * (w + h / 2 * a1)) + uu
                a4 = -math.sin(th + h * (w + h / 2 * a2)) + uu
                th, w = th + h / 6 * (w + 2 * (w + h / 2 * a1) + 2 * (w + h / 2 * a2) + (w + h * a3)), w + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            return np.array([th, w])
    gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
    g = lambda x: np.array([Rg ** 2 - gd(x) ** 2]); miss = lambda X: np.maximum(gd(X) - Rg, 0); flc = count_flow(fl)
    S = A.SysN(flc, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, [(np.pi + Rg * 0.999 * np.cos(a), Rg * 0.999 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)], per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
    rng = np.random.default_rng(1); Q = []
    while len(Q) < NQ:
        x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
        if gd(x) > 1.0: Q.append(x)
else:
    from src.atlas6.manip2dyn import flow4, clearance
    obst = SYS == 'manip_obst'; NB, NF, K = (2400, 400, 3) if obst else (600, 300, 3)
    tau = 0.4; Rq = 0.3; Rw = 0.5; WM = 3.0; rho = 0.15
    wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    fl = lambda P, s, t: flow4(P, s, t, dt_max=0.05)
    if os.environ.get('SCALAR'):                       # прототип: точка — чистый Python (flow4_scalar.py), батч — flow4
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from flow4_scalar import flow4s
        fl = lambda P, s, t: flow4s(P, s, t, dt_max=0.05)
    ing = lambda P: (np.max(np.abs(wr(P[..., :2])), -1) < Rq) & (np.max(np.abs(P[..., 2:]), -1) < Rw)
    g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rw ** 2 - x[2] ** 2, Rw ** 2 - x[3] ** 2])
    miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :2])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 2:]), -1) - Rw, 0)
    rng = np.random.default_rng(0); seeds = []
    for _ in range(32):
        v = rng.uniform(-1, 1, 4); v = v / np.max(np.abs(v)) * 0.99; seeds.append(v * np.array([Rq, Rq, Rw, Rw]))
    OB = [((1.2, 0.8), 0.3), ((-0.5, 1.4), 0.25)]
    clear = (lambda P: clearance(P, OB) - 0.02) if obst else None; blk = (lambda P: clearance(np.asarray(P), OB) < 0) if obst else None
    flc = count_flow(fl)
    S = A.SysN(flc, 4, (np.pi, np.pi, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 0, 0), ok=lambda p: np.max(np.abs(p[2:])) <= WM, blocked=blk)
    Q = [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(14)]
    Q = [x for x in Q if not (obst and blk(x))][:NQ]
if len(sys.argv) > 5: NB, NF, K = [int(a) for a in sys.argv[3:6]]

if os.environ.get('FAST'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import refine_fast as RF; RF.minimize = minimize_c; C.refine = RF.refine_seq if os.environ['FAST'] == 'seq' else RF.refine_fast
timed(A, '_tree', 'tree_build'); timed(A, 'tree_query', 'kd_query'); timed(C, '_tree', 'tree_fwd'); timed(C, 'tree_query', 'kd_query')
timed(C, 'candidates', 'candidates'); timed(C, 'refine', 'refine')
t = time.perf_counter(); back = A.build_back(S, tau, NB, rho); T['build_back'] = time.perf_counter() - t
T0 = dict(T); N0 = dict(N); T.clear(); N.clear()
pr = cProfile.Profile(); res = []
for x in Q:
    t = time.perf_counter(); V = A.replay_value_fast(S, tau, x, NB, NF, rho, rho, back=back)[0]; tv = time.perf_counter() - t
    t = time.perf_counter(); pr.enable(); b = C.corridor_query(S, flc, x, back, tau, NF, rho, g, miss, K=K, clear=clear); pr.disable()
    res.append(dict(V=round(float(V), 3), T=None if b is None else round(b[0], 3), nseg=None if b is None else len(b[1]), t_replay=round(tv, 2), t_corr=round(time.perf_counter() - t, 2)))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14); print(s.getvalue()[-3500:], file=sys.stderr)
out = dict(fast=os.environ.get('FAST'), scalar=bool(os.environ.get('SCALAR')), host=platform.node(), numpy=np.__version__, scipy=scipy.__version__, load=os.getloadavg()[0], sys=SYS, NB=NB, NF=NF, K=K, NQ=len(Q),
           build=dict(t=round(T0['build_back'], 2), tree=round(T0.get('tree_build', 0), 2), flow_point=round(T0.get('flow_point', 0), 2), n_flow_point=N0.get('flow_point', 0)),
           maxit=os.environ.get('MAXIT'), nit_ok_pct=[int(np.percentile(NIT0, q)) for q in (50, 90, 99, 100)] if NIT0 else None, slsqp_by_status={k: [v[0], round(v[1], 2), v[2]] for k, v in SL.items()}, queries=res, stages={k: round(v, 2) for k, v in T.items()}, counts=N)
print(json.dumps(out, ensure_ascii=False))
