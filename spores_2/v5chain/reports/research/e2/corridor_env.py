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

def setup():
    e = common.make(name); d = e.n
    if os.environ.get('STARTS'):
        a = np.load(os.path.expanduser(os.environ['STARTS'])); e.X0 = a[:, :4].copy(); e.Tref = a[:, 6].copy(); Q = e.X0
        e.keep = np.abs(np.c_[Q[:, 0] + Q[:, 2] * np.abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * np.abs(Q[:, 3]) / 2]).max(1) <= 2.5
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
    WM = float(np.min(e.hi[per == 0])) if (per == 0).any() else 3.
    ok = lambda p: not bool(e.out_of_box(np.asarray(p, float)[None])[0])
    S = SysN(flow, len(US), tuple(np.where(per > 0, np.pi, np.maximum(np.abs(e.hi), 1.))), ing, seeds, per=tuple(per), ok=ok)
    wl = (tuple(int(i) for i in np.nonzero(per == 0)[0]), WM) if WL else None
    return e, S, US, flow, g, miss, wl

def init_w(pk):
    global E, S, US, FL, G, MISS, WLIM, BACK
    E, S, US, FL, G, MISS, WLIM = setup(); BACK = pickle.load(open(pk, 'rb'))
def work(i):
    E.nf = 0; t1 = time.perf_counter()
    b = corridor_query(S, FL, E.X0[i], BACK, tau, NF, rho, G, MISS, K=K, tries=3, kn=8, neigh=2, nbest=1, wlim=WLIM); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path')
    us = [US[s] for s in b[1]]; Tv, why = common.verify(E, E.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, nseg=len(us))

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
