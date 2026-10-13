"""CA4-6D+2 диска (линия A, w28): последовательные цели на 3 звеньях 6D + g .3 + 2 диска (all_systems 8d); цели — свободные от дисков позы. K переходов между случайными целями (окно ±(.3,.3,.3,.6,.6,.6) вокруг (q*,0)), старт перехода k = центр цели k-1 в покое (w=0).
python m3gdg_goals.py ref|fast|sst|spores [K] [--procs N]  → data/m3gdg_{ref,ocpfast,sst,spores}.json; PPO с целью на входе не делали (обучение > 1 ч).
спор: на цель — build_back (заготовка цели), на переход — запрос; OCP: multi = эталон (8 ветвей×4 затравки), fast = ближайшая ветвь×2 затравки; SST — на переход, бюджеты .3/1/3/10 с."""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
PI = np.pi; wr = lambda x: (x + PI) % (2 * PI) - PI
import manip3g_dyn as G3
OB = [((1.0, -1.5), 0.3), ((-1.0, -1.0), 0.3)]
def goals(K):
    rng = np.random.default_rng(int(os.environ.get('GSEED', 11))); g = []
    while len(g) < K + 1:
        q = rng.uniform(-PI, PI, 3)
        if G3.clearance(q, OB) > .1: g.append(q)
    return np.array(g)
def transitions(K): g = goals(K); return [(g[k], g[k + 1]) for k in range(K)]
class MGoal(envs_6d.M3G):
    def __init__(s, goal, start):
        envs_6d.M3G.__init__(s, 1); s.C3 = np.r_[np.asarray(goal, float), 0., 0., 0.]; s.X0 = np.array([[*start, 0., 0., 0.]]); s.Tref = np.array([np.nan]); s.keep = np.ones(1, bool); s.nf = 0
    def out_of_box(s, Y): Y = np.asarray(Y, float); return envs_6d.M3G.out_of_box(s, Y) | (G3.clearance(Y[..., :3], OB) < 0)      # коробка |w| и столкновение с дисками
def ocp_work(a):
    import ocp_m3go_lib as L
    k, st, g, var = a; e = MGoal(g, st); t0 = time.perf_counter()
    best, res = L.solve_query(e.X0[0], g, 50, 4, 4 if var == 'multi' else 2, seed=k, solver=SOL, nearest=(var == 'fast')); dt = time.perf_counter() - t0
    if best is None: return dict(k=k, T=None, t=dt, why='no_sol')
    T, U, br = best; Tv, why = common.verify(e, e.X0[0], U, np.full(len(U), T / len(U))); return dict(k=k, T=Tv, t=dt, why=why)
SOL = None
BUDS = [.3, 1., 3., 10.]
def sst_work(a):
    import sst_own
    k, st, g = a; e = MGoal(g, st); BUD = [.3, 1., 3., 10.]; snaps = sst_own.sst(e, e.X0[0], BUD, sel=.2, prune=.1, seed=1000 + k); out = []
    for s in snaps:
        r = dict(budget=s['budget'], t=s['t'], has=s['has'])
        if s['has']: T, why = common.verify(e, e.X0[0], s['us'], s['dts']); r.update(T=T, why=why)
        out.append(r)
    return dict(k=k, runs=out)
def hyb_refine(HS, L, x0, gl, seq, dts, fl, US_, N=50):
    """HYB (w29): путь спор → тёплый старт полного OCP (свободное T, N=50, зазор до дисков в узлах). → (U[N,3], T) либо None, если не сошлось/не прошло проверку."""
    S, lbx, ubx, lbg, ubg = HS; seq = list(seq); dts = list(dts); T = float(sum(dts)); ts = np.linspace(0, T, N + 1); ce = np.cumsum(dts)
    x = np.array(x0, float); X = [x.copy()]; seg = 0; t = 0.
    for i in range(1, N + 1):
        while t < ts[i] - 1e-12:
            step = min(ts[i], ce[seg]) - t; x = fl(x, seq[seg], step); t += step
            if t >= ce[seg] - 1e-12 and seg < len(seq) - 1: seg += 1
        X.append(x.copy())
    X = np.array(X); tm = (ts[:-1] + ts[1:]) / 2; Ug = np.array([US_[seq[min(int(np.searchsorted(ce, v)), len(seq) - 1)]] for v in tm], float)
    C3 = np.asarray(gl, float); tgt = C3 + 2 * np.pi * np.round((X[-1][:3] - C3) / (2 * np.pi)); P = np.concatenate([x0, tgt, np.zeros(3)])
    r = S(x0=np.concatenate([X.ravel(), Ug.ravel(), [T]]), p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
    if st not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): return None, st
    wv = np.array(r['x']).ravel(); To = wv[-1]; U = wv[6 * (N + 1):6 * (N + 1) + 3 * N].reshape(N, 3); xe, wmax, mc = L.simulate(x0, U, To)
    ok = np.all(np.abs(L.wr(xe[:3] - C3)) < L.Rq + 1e-3) and np.all(np.abs(xe[3:]) < L.Rw + 1e-3) and wmax <= L.WM + 1e-2 and mc >= 0
    return ((U, To) if ok else None), st
def run_spores(K):
    sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
    import refine_ca, tree_fast
    US_ = np.array(G3.LAYERS6)
    from src.atlas6.adaptive_nd import SysN, build_back
    from src.atlas6.corridor_nd import corridor_query
    tau, Rq, Rw, WM, rho = .4, .3, .6, 3., .15; NB, NF, KK = int(os.environ.get('NBG', 24000)), int(os.environ.get('NFG', 6000)), 3
    DG = {}; HS = None
    if os.environ.get('HYB'):
        import ocp_m3go_lib as _L; t0h = time.perf_counter(); HS = _L.make_solver(50, 4); print('HYB: OCP-решатель собран за %.1f с' % (time.perf_counter() - t0h), flush=True)
    if os.environ.get('DIAG'):
        from src.atlas6 import corridor_nd as _CN; _c0 = _CN.candidates
        def _cw(*a, **kw):
            C, nh = _c0(*a, **kw); DG['cand'] = len(C); DG['hits'] = nh; return C, nh
        _CN.candidates = _cw
    CAR = refine_ca.install('m3gd', ms=10, maxit=60, prebuild=range(1, 14), wl=True); tree_fast.install(); flow_all = tree_fast.make_flow_all('m3gd', US_); fl = lambda P, s, t: G3.flow(P, s, t, dt_max=.05, g=.3)
    if os.environ.get('BATCHFAST'): fl = tree_fast.make_flow_batch('m3gd', US_, fl)      # пакетный поток стыковки (t=±.2), остальное — исходный
    OBS = lambda P: G3.clearance(np.asarray(P)[..., :3], OB); blk = lambda P: OBS(P) < 0; clear = lambda P: OBS(P) - .02; WLIM = ((3, 4, 5), WM)
    res = []; Tr = transitions(K)
    for k, (st, gl) in enumerate(Tr):
        if os.environ.get('ONLYK') and k not in [int(v) for v in os.environ['ONLYK'].split(',')]: continue      # диагностика (w29): только указанные переходы
        e = MGoal(gl, st); C3 = e.C3; CAR.C3 = C3[:3].copy(); t0 = time.perf_counter()
        ing = lambda P: (np.max(np.abs(wr(P[..., :3] - C3[:3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
        g = lambda x: np.array([Rq ** 2 - wr(x[0] - C3[0]) ** 2, Rq ** 2 - wr(x[1] - C3[1]) ** 2, Rq ** 2 - wr(x[2] - C3[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
        miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3] - C3[:3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
        rng = np.random.default_rng(0); seeds = []
        for _ in range(32): v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C3 + v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
        S = SysN(fl, 8, (PI, PI, PI, WM, WM, WM), ing, seeds, per=(2 * PI, 2 * PI, 2 * PI, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM, blocked=blk); S.flow_all = flow_all
        back = build_back(S, tau, NB, rho); tb = time.perf_counter() - t0; t1 = time.perf_counter()
        b = corridor_query(S, fl, e.X0[0], back, tau, NF, rho, g, miss, K=KK, tries=int(os.environ.get('TRIESG', 3)), kn=8, neigh=int(os.environ.get('NEIGH', 1)), clear=clear, wlim=WLIM)
        if b is None:                                            # откат на штатный SLSQP-refine (редко; время входит в запрос)
            from src.atlas6 import corridor_nd as _C; _C.refine = CAR.orig
            try: b = corridor_query(S, fl, e.X0[0], back, tau, NF * int(os.environ.get('RETRYNF', 1)), rho, g, miss, K=KK, tries=int(os.environ.get('TRIESG', 3)), kn=8, neigh=int(os.environ.get('NEIGH', 1)), clear=clear, wlim=WLIM)
            finally: _C.refine = CAR.refine
        dq = time.perf_counter() - t1
        if os.environ.get('DIAG'): print('DIAG k=%d refine-stat=%s ipopt-status=%s nh/nm=%s' % (k, CAR.stat, CAR.st, DG), flush=True)
        if b is None: r = dict(k=k, T=None, why='no_path', t_build=tb, t_query=dq)
        else: Tv, why = common.verify(e, e.X0[0], [US_[s] for s in b[1]], list(b[2])); r = dict(k=k, T=Tv, why=why, t_build=tb, t_query=dq, nseg=len(b[1]))
        if os.environ.get('VIZSEQ') and b is not None: r.update(us=[US_[s_].tolist() for s_ in b[1]], dts=[float(v) for v in b[2]])      # линия D (viz): план для воспроизведения
        if HS is not None and b is not None and r.get('T'):
            th0 = time.perf_counter(); res_h, sth = hyb_refine(HS, _L, e.X0[0], gl, b[1], b[2], fl, US_); th = time.perf_counter() - th0
            r.update(T_sp=r['T'], t_hyb=th, hyb_status=sth, hyb_ok=False)
            if res_h is not None:
                Uh, Th = res_h; Tvh, whyh = common.verify(e, e.X0[0], Uh, np.full(len(Uh), Th / len(Uh))); r['T_hyb'] = Tvh; r['hyb_why'] = whyh
                if os.environ.get('VIZSEQ') and whyh == 'ok' and Tvh < r['T_sp']: r.update(us=Uh.tolist(), dts=[float(Th) / len(Uh)] * len(Uh))      # линия D: HYB-траектория
                if whyh == 'ok' and Tvh < r['T']: r.update(T=Tvh, hyb_ok=True)
            r['t_query'] = dq + th
        res.append(r); print(k, r, flush=True)
    return res
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd'); p.add_argument('K', type=int, nargs='?', default=20); p.add_argument('--procs', type=int, default=10); a = p.parse_args(); K = a.K; Tr = transitions(K); os.makedirs(common.DATA, exist_ok=True)
    if a.cmd in ('ref', 'fast'):
        import ocp_m3go_lib as L; from multiprocessing import get_context
        t0 = time.perf_counter(); SOL = L.make_solver(50, 4); tbuild = time.perf_counter() - t0
        with get_context('fork').Pool(a.procs) as pool: R = pool.map(ocp_work, [(k, st, g, 'multi' if a.cmd == 'ref' else 'fast') for k, (st, g) in enumerate(Tr)], chunksize=1)
        json.dump(dict(cmd=a.cmd, t_build=tbuild, K=K, res=R, goals=goals(K).tolist()), open(os.path.join(common.DATA, 'm3gdg_%s%s.json' % ('ref' if a.cmd == 'ref' else 'ocpfast', os.environ.get('OUTSFX', ''))), 'w'), indent=1, default=float)
    elif a.cmd == 'sst':
        from multiprocessing import Pool
        with Pool(a.procs) as pool: R = pool.map(sst_work, [(k, st, g) for k, (st, g) in enumerate(Tr)], chunksize=1)
        json.dump(dict(K=K, res=R), open(os.path.join(common.DATA, 'm3gdg_sst%s.json' % os.environ.get('OUTSFX', '')), 'w'), indent=1, default=float)
    else:
        R = run_spores(K); json.dump(dict(K=K, res=R), open(os.path.join(common.DATA, 'm3gdg_spores%s.json' % os.environ.get('TAG', '')), 'w'), indent=1, default=float)
    print('M3GDG', a.cmd, 'done', flush=True)
