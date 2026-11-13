"""e2 (линия E): коридор v6 (обратное дерево NB + прямое NF + SLSQP-доводка, K топологий) на средах di4 / dp1 / manip (4D) — те же старты/эталоны, что у графа.
Конфиг как у SPD6 (bench/spores_6d.py): tau .4, rho .15, K 3, TRIES 3, kn 8, neigh 2, NB 12000, NF 3000 (env NB/NF/KK). Отличие от table_m3*: доводка — штатный SLSQP corridor_nd.refine
(CasADi/IPOPT-refine refine_ca есть только для рук m3d/m3g/m8 — для 4D-сред динамики fca нет). Действия — 4 угла куба u (как слои 2^m у рук). Окно цели: half-WIN (WIN=.01).
Запуск: python corridor_env.py <di4|dp1|manip> [tag]   env: NB NF KK PROCS STARTS(npy di4_ref) WL=1 (|v|<=WM вдоль пути) ; Windows: spawn Pool (back — pickle)."""
import os, sys, time, json, pickle
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.expanduser(os.environ.get('W43', '~/spore_v5/w43')); sys.path[:0] = [os.path.join(HERE, 'spores_2', 'bench'), os.path.join(HERE, 'spores_2', 'v6')]
import numpy as np, common
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
import multiprocessing as mp
name = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else ''
NB, NF, K = int(os.environ.get('NB', 12000)), int(os.environ.get('NF', 3000)), int(os.environ.get('KK', 3)); tau = .4; rho = .15; WIN = float(os.environ.get('WIN', .01)); WL = os.environ.get('WL') == '1'

CAR = None
def setup():
    e = common.make(name); d = e.n
    if os.environ.get('STARTS'):
        a = np.load(os.path.expanduser(os.environ['STARTS'])); e.X0 = a[:, :4].copy(); e.Tref = a[:, 6].copy(); Q = e.X0
        e.keep = np.abs(np.c_[Q[:, 0] + Q[:, 2] * np.abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * np.abs(Q[:, 3]) / 2]).max(1) <= 2.5
    if os.environ.get('STARTSNPY') and name == 'dp1':   # e2: новые старты dp1 (npy: x(4) обычные координаты, T) — преобразование как в common.DP1
        a = np.load(os.path.expanduser(os.environ['STARTSNPY'])); e.X0 = common.wrap(a[:, :4] - np.array([np.pi / 2, 0, 0, 0]), e.period); e.Tref = a[:, 4].copy(); e.keep = np.isfinite(e.Tref); e.Tref = np.where(e.keep, e.Tref, 8.)
        if os.environ.get('SEEDN') or 'new20' in os.environ['STARTSNPY']:  # new20 ⇒ SEEDN по умолчанию 777   # точные старты (в npy округлены до 1e-4 — на границе неустойчивой системы важно): повтор ref_e2.starts()
            rng = np.random.default_rng(int(os.environ.get('SEEDN', 777)))
            for _ in range(32): rng.uniform(-1, 1, 4)
            Xe = np.array([np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(len(a))]); assert np.abs(Xe - a[:, :4]).max() < 1e-3; e.X0 = common.wrap(Xe - np.array([np.pi / 2, 0, 0, 0]), e.period)
    US = np.array([[a_, b_] for a_ in (-1., 1.) for b_ in (-1., 1.)]) * np.array(e.umax)
    hw = e.half - WIN; per = np.asarray(e.period, float)
    def flow(P, s, t, dt_max=None):
        P = np.asarray(P, float); n = max(1, int(np.ceil(abs(t) / (.02 if name != 'di4' else 10.)))); h = t / n; Z = P; u = np.broadcast_to(US[s], Z.shape[:-1] + (e.m,))
        for _ in range(n): Z = e.step(Z, u, h)
        return Z
    wr = lambda x: common.wrap(np.asarray(x, float), per)
    ing = lambda P: (np.abs(wr(P)) <= hw).all(-1)
    g = lambda x: hw ** 2 - wr(x) ** 2
    miss = lambda X: np.maximum(np.abs(wr(X)) - hw, 0).max(-1)
    rng = np.random.default_rng(0); seeds = rng.uniform(-hw, hw, (32, d))
    BOXF = float(os.environ.get('BOXF', 1.0)); vi_ = np.nonzero(per == 0)[0]
    WM = float(np.min(e.hi[per == 0])) / BOXF if (per == 0).any() else 3.   # BOXF>1: запас коробки (предел |v| = hi/BOXF и в узлах дерева, и вдоль пути)
    ok = lambda p: (not bool(e.out_of_box(np.asarray(p, float)[None])[0])) and (BOXF == 1.0 or bool((np.abs(np.asarray(p, float)[vi_]) <= WM).all()))
    S = SysN(flow, len(US), tuple(np.where(per > 0, np.pi, np.maximum(np.abs(e.hi), 1.))), ing, seeds, per=tuple(per), ok=ok)
    NSW = int(os.environ.get('NSW', 16)); WMARG = float(os.environ.get('WMARG', 0.))   # правка (ii) d4: NSW точек |w| на отрезок вместо 16, АБСОЛЮТНЫЙ запас WMARG на предел (3.01-.05)
    if NSW != 16:
        from src.atlas6 import corridor_nd as _C
        if not getattr(_C, '_ns_patched', False):
            _ref0 = _C.refine; _C.refine = lambda flow, x0, path, g, tries=3, seed=0, clear=None, ns=16, wlim=None, _r=_ref0: _r(flow, x0, path, g, tries=tries, seed=seed, clear=clear, ns=NSW, wlim=wlim); _C._ns_patched = True
    if os.environ.get('VEC'):   # c14 tree_vec: векторизованное прямое дерево (порядок раскрытия прежний); ok_batch — та же коробка, что ok
        import tree_vec, sys as _s
        S.ok_batch = lambda E_: ~(((np.asarray(E_) < e.lo) | (np.asarray(E_) > e.hi)) & (per == 0)).any(-1)
        tree_vec.install(_s.modules['src.atlas6.adaptive_nd'], _s.modules['src.atlas6.corridor_nd'])
    wl = (tuple(int(i) for i in np.nonzero(per == 0)[0]), WM - WMARG) if WL else None
    global CAR
    if os.environ.get('CA4'):   # IPOPT-доводчик (refine_ca4.py) вместо SLSQP: плотная проверка |w| в MS точках, предел WM-WMARG
        import refine_ca4; CAR = refine_ca4.install(name, ms=int(os.environ.get('CAMS', 64)), maxit=int(os.environ.get('CAIT', 60)), wmc=WM - WMARG)
    if os.environ.get('NQSUB'): _ix = [int(v) for v in os.environ['NQSUB'].split(',')]; e.X0 = e.X0[_ix]; e.Tref = np.asarray(e.Tref)[_ix]; e.keep = np.asarray(e.keep)[_ix]   # e3: подвыборка стартов для проб
    return e, S, US, flow, g, miss, wl

def init_w(pk):
    global E, S, US, FL, G, MISS, WLIM, BACK
    E, S, US, FL, G, MISS, WLIM = setup(); BACK = pickle.load(open(pk, 'rb'))
def work(i):
    E.nf = 0; t1 = time.perf_counter()
    if CAR is not None: CAR.stat.clear(); CAR.stat.update(n=0, t=0., it=0)
    b = corridor_query(S, FL, E.X0[i], BACK, tau, NF, rho, G, MISS, K=K, tries=3, kn=8, neigh=2, nbest=1, wlim=WLIM); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path', cstat=(dict(CAR.stat) if CAR is not None else None), st=(dict(CAR.st) if CAR is not None else None))
    us = [US[s] for s in b[1]]; Tv, why = common.verify(E, E.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, nseg=len(us), cstat=(dict(CAR.stat) if CAR is not None else None), st=(dict(CAR.st) if CAR is not None else None))

if __name__ == '__main__':
    e, S, US, FL, G, MISS, WLIM = setup(); t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с, узлов %d' % (NB, t_build, len(back[0])), flush=True)
    pk = os.path.join(HERE, 'out', 'back_%s%s.pkl' % (name, tag)); os.makedirs(os.path.dirname(pk), exist_ok=True); pickle.dump(back, open(pk, 'wb'))
    t0 = time.perf_counter(); n = len(e.X0)
    res = None
    for att in range(12):   # Windows/ssh: CreateProcess иногда даёт WinError 5 — повторять
        try:
            with mp.get_context('spawn').Pool(min(int(os.environ.get('PROCS', 16)), n), initializer=init_w, initargs=(pk,)) as p: res = p.map(work, range(n), chunksize=1)
            break
        except PermissionError as ex: print('pool retry', att, ex, flush=True); time.sleep(3)
    assert res is not None
    wall = time.perf_counter() - t0; Ts = [r['T_verify'] for r in res]; tq = np.array([r['t_query'] for r in res]); s = common.summarize(e, Ts)
    keep = e.keep; viol = int(sum(1 for r, k in zip(res, keep) if k and r['why'] == 'box')); nopath = int(sum(1 for r, k in zip(res, keep) if k and r['why'] == 'no_path'))
    s.update(env=name, method='коридор v6 NB%d NF%d K%d%s (SLSQP)' % (NB, NF, K, ' wlim' if WL else ''), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_wall_parallel=wall, box_viol=viol, no_path=nopath,
             per_start=res, Tref=e.Tref.tolist(), weight=common.deep_weight(back), nodes_back=int(len(back[0])), procs=min(int(os.environ.get('PROCS', 16)), n))
    os.makedirs('out', exist_ok=True); json.dump(s, open('out/corr_%s_%d_%d%s.json' % (name, NB, NF, tag), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
