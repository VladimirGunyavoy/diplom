"""линия A: споры 8D (4 звена + g, коридор v6: обратное дерево NB + прямое NF + SLSQP, эвристика back_heuristic) на m8. Заготовка = build_back, запрос = corridor_query (1 ядро/запрос). python spores_8d.py m8 NB NF K [tag] [wlim]"""

import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))                  # после common: тот добавляет v8 (тоже пакет src)
from src.atlas6.adaptive_nd import SysN, build_back, back_heuristic
from src.atlas6.corridor_nd import corridor_query
import multiprocessing as mp
name = 'm8'; NB, NF, K = [int(a) for a in sys.argv[2:5]]; tag = sys.argv[5] if len(sys.argv) > 5 else ''; WL = len(sys.argv) > 6 and sys.argv[6] == 'wlim'
e = common.make(name); n = 4; tau = .4; Rq, Rw, WM = .3, .6, 3.; rho = .15; C = np.array(e.C3, float)
import manip3g_dyn as G3
fl = lambda P, s, t: G3.flow(P, s, t, dt_max=.02, n=n, g=e.G)
US_ = np.array([[1.0 if (s >> i) & 1 else -1.0 for i in range(n)] for s in range(2 ** n)])
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.max(np.abs(wr(P[..., :n] - C[:n])), -1) < Rq) & (np.max(np.abs(P[..., n:]), -1) < Rw)
g = lambda x: np.concatenate([Rq ** 2 - wr(x[:n] - C[:n]) ** 2, Rw ** 2 - x[n:] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :n] - C[:n])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., n:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []; R8 = np.array([Rq] * n + [Rw] * n)
for _ in range(32):
    v = rng.uniform(-1, 1, 2 * n); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C + v * R8)
S = SysN(fl, 2 ** n, (np.pi,) * n + (WM,) * n, ing, seeds, per=(2 * np.pi,) * n + (0,) * n, ok=lambda p: np.max(np.abs(p[n:])) <= WM)
if os.environ.get('CAREF'):
    import refine_ca; t0 = time.perf_counter(); CAR = refine_ca.install(name, ms=int(os.environ.get('CAMS', 10)), maxit=int(os.environ.get('CAIT', 60)), prebuild=range(1, int(os.environ.get('CAPRE', 12))), wl=WL); print('refine_ca: сборка солверов %.1f с' % (time.perf_counter() - t0), flush=True)
if os.environ.get('TREEFAST'):
    import tree_fast; tree_fast.install(); t0 = time.perf_counter(); S.flow_all = tree_fast.make_flow_all(name, US_); print('tree_fast: компиляция потока %.1f с' % (time.perf_counter() - t0), flush=True)
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); HF = back_heuristic(S, back, wh=None); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с' % (NB, t_build), flush=True)
def work(i):
    e.nf = 0; t1 = time.perf_counter(); b = corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=3, kn=8, neigh=int(os.environ.get('NEIGH', 2)), hfun=HF, wlim=(tuple(range(n, 2 * n)), WM) if WL else None); dt = time.perf_counter() - t1
    if b is None and os.environ.get('NFRETRY'):       # не нашли путь — один повтор с большим прямым деревом (только провалы платят)
        b = corridor_query(S, fl, e.X0[i], back, tau, int(os.environ['NFRETRY']), rho, g, miss, K=K, tries=3, kn=8, neigh=int(os.environ.get('NEIGH', 2)), hfun=HF, wlim=(tuple(range(n, 2 * n)), WM) if WL else None); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path')
    us = [US_[s] for s in b[1]]; Tv, why = common.verify(e, e.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, nseg=len(us))
if __name__ == '__main__':
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 16)), len(e.X0))) as p: res = p.map(work, range(len(e.X0)), chunksize=1)
    wall = time.perf_counter() - t0; Ts = [r['T_verify'] for r in res]; tq = np.array([r['t_query'] for r in res]); s = common.summarize(e, Ts)
    s.update(env=name, method='споры 6D коридор v6 NB%d NF%d K%d%s' % (NB, NF, K, ' wlim' if WL else ''), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(tq.sum()), t_wall_parallel=wall,
             hw='CPU 1 ядро/запрос, numpy+math-поток', per_start=res, Tref=e.Tref.tolist(), weight=common.deep_weight(back), nodes_back=int(len(back[0])))
    os.makedirs(common.DATA, exist_ok=True); json.dump(s, open(os.path.join(common.DATA, '%s_spores8d%s.json' % (name, tag)), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
