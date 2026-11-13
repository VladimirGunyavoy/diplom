"""c13: КОРИДОР v6 (SPD6, bench/spores_6d.py) на ДИ 4D/6D — тот же алгоритм/конфиг (tau .4, rho .15, NB 12000, NF 1000, K 3, NEIGH 0, SLSQP-refine — CAREF/IPOPT только для m3d),
среда: k независимых осей (x_i,v_i), ẍ_i=u_i, |u_i|≤1; слои = 2^k углов (bang-bang), поток точный; цель — ящик ±.35 по всем; проверка common.verify.
python spores_di.py di4|di6 NB NF K tag STARTS(npy, x(d)…T_ref в последнем столбце)  → c13/out_spd6/{env}_spd6_{tag}.json ; env: PROCS (≤ потоков), NEIGH, TAU, RHO."""
import os, sys, time, json, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1')
BENCH = os.path.expanduser('~/spore_v5/a31/spores_2/bench'); sys.path.insert(0, BENCH)
import numpy as np, common
sys.path.insert(0, os.path.join(BENCH, '..', 'v6'))
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
import multiprocessing as mp
name = sys.argv[1]; NB, NF, K = [int(a) for a in sys.argv[2:5]]; tag = sys.argv[5]; STARTS = os.path.expanduser(sys.argv[6])
dim = int(name[2:]); k = dim // 2; tau = float(os.environ.get('TAU', .4)); rho = float(os.environ.get('RHO', .15)); R = .35
if name == 'di4': e = common.make('di4')
else:
    class DIX(common.DI4):
        n = dim; m = k; period = np.zeros(dim); half = np.full(dim, R); umin = -np.ones(k); umax = np.ones(k)
        lo = np.array([-3.5] * k + [-4.] * k); hi = -lo; fl_lo = np.array([-2.5] * k + [-2.] * k); fl_hi = -fl_lo
        def __init__(s): s.nf = 0
        def f(s, Y, u): s.nf += 1; return np.concatenate([Y[..., k:], np.broadcast_to(u, Y[..., :k].shape)], -1)
        def step(s, Y, u, h):
            s.nf += 1; h = np.asarray(h, float)[..., None] if np.ndim(h) else h; a = np.broadcast_to(u, Y[..., :k].shape); return np.concatenate([Y[..., :k] + Y[..., k:] * h + a * h * h / 2, Y[..., k:] + a * h], -1)
    e = DIX(); e.name = name
a_ = np.load(STARTS); e.X0 = a_[:, :dim].copy(); e.Tref = a_[:, -1].copy(); Q = e.X0
e.keep = np.abs(Q[:, :k] + Q[:, k:] * np.abs(Q[:, k:]) / 2).max(1) <= 2.5      # как tvg_vt/DI4: остановочный путь в поле
import hashlib
print('HEADER: starts+эталон %s md5=%s (n=%d, keep=%d); проверяльщик common.verify h=.01 строгий, tfac 3; окно цели ±%.2f (в refine запас MG=%s); коробка verify lo/hi %s/%s; флаги NB=%d NF=%d K=%d tau=%g rho=%g NEIGH=%s SLSQP-refine (не IPOPT)' % (STARTS, hashlib.md5(open(STARTS, 'rb').read()).hexdigest()[:10], len(e.X0), e.keep.sum(), R, os.environ.get('MG', .004), e.lo.tolist(), e.hi.tolist(), NB, NF, K, tau, rho, os.environ.get('NEIGH', 0)), flush=True)
US_ = np.array(list(itertools.product((-1., 1.), repeat=k)))
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); t = np.asarray(t, float)[..., None]; a = US_[s]
    return np.concatenate([P[..., :k] + P[..., k:] * t + a * t * t / 2, P[..., k:] + a * t], -1)
ing = lambda P: np.max(np.abs(P), -1) < R
MG = float(os.environ.get('MG', .004)); Rg = R - MG   # запас у цели (как «IPOPT-refine, запас .004» в строке m3d): refine садится на границу, verify строгий <
g = lambda x: np.array([Rg ** 2 - xi ** 2 for xi in x])
miss = lambda X: np.maximum(np.max(np.abs(X), -1) - Rg, 0)
rng = np.random.default_rng(0); seeds = [rng.uniform(-1, 1, dim) / 1.0 for _ in range(32)]
seeds = [v / np.max(np.abs(v)) * .99 * R for v in seeds]
hi = np.array(e.hi)
S = SysN(flow, 2 ** k, np.asarray(e.hi, float), ing, seeds, per=None, ok=lambda p: bool(np.all(np.abs(p) <= hi)))
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с' % (NB, t_build), flush=True)
def work(i):
    if not e.keep[i]: return dict(i=i, skip=True)
    t1 = time.perf_counter(); b = corridor_query(S, flow, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=int(os.environ.get('TRIES', 3)), kn=8, neigh=int(os.environ.get('NEIGH', 0)), nbest=1); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path')
    us = [US_[s] for s in b[1]]; Tv, why = common.verify(e, e.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, nseg=len(us))
if __name__ == '__main__':
    NS = int(os.environ.get('NSTART', len(e.X0))); t0 = time.perf_counter()
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, range(NS), chunksize=1)
    wall = time.perf_counter() - t0; rr = [r for r in res if not r.get('skip')]; T = np.array([np.nan if r['T_verify'] is None else r['T_verify'] for r in rr]); ref = np.array([e.Tref[r['i']] for r in rr])
    ok = np.isfinite(T); ratio = T[ok] / ref[ok]; tq = np.array([r['t_query'] for r in rr]); why = {w: sum(1 for r in rr if r['why'] == w) for w in set(r['why'] for r in rr)}
    s = dict(env=name, NB=NB, NF=NF, K=K, tau=tau, rho=rho, n=len(rr), reach=int(ok.sum()), why=why, slow3=int((ratio > 3).sum()), med=float(np.median(ratio)) if ok.any() else None, p90=float(np.percentile(ratio, 90)) if ok.any() else None, max=float(ratio.max()) if ok.any() else None,
             t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), wall=wall, nodes_back=int(len(back[0])), weight=common.deep_weight(back), per_start=res)
    o = os.path.expanduser('~/spore_v5/c13/out_spd6'); os.makedirs(o, exist_ok=True); json.dump(s, open('%s/%s_spd6_%s.json' % (o, name, tag), 'w'), indent=1, default=float)
    print({k_: v for k_, v in s.items() if k_ != 'per_start'}, flush=True)
