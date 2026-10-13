"""линия A: споры 6D (коридор v6: обратное адаптивное дерево NB + прямое NF + SLSQP) на среде m3d (envs_6d). Заготовка = build_back (раз на цель), запрос = corridor_query (1 ядро на запрос, параллельно по стартам).
python spores_6d.py m3d NB NF K [tag] [wlim]   → data/{env}_spores6d{tag}.json   (aida: ~/spore_v5/a27/spores_2/bench)"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))                  # после common: тот добавляет v8 (тоже пакет src)
import manip3d_dyn as M3
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
import multiprocessing as mp
name = sys.argv[1]; NB, NF, K = [int(a) for a in sys.argv[2:5]]; tag = sys.argv[5] if len(sys.argv) > 5 else ''; WL = len(sys.argv) > 6 and sys.argv[6] == 'wlim'
e = common.make(name); tau = .4; Rq, Rw, WM = .3, .6, 3.; rho = .15; C3 = np.zeros(6)
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: M3.flow(P, s, t, dt_max=.05)
ing = lambda P: (np.max(np.abs(wr(P[..., :3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C3 + v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
S = SysN(fl, 8, (np.pi, np.pi, np.pi, WM, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM)
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с' % (NB, t_build), flush=True)
def work(i):
    e.nf = 0; t1 = time.perf_counter(); b = corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=3, kn=8, wlim=((3, 4, 5), WM) if WL else None); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path')
    us = [M3.US[s] for s in b[1]]; Tv, why = common.verify(e, e.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, nseg=len(us))
if __name__ == '__main__':
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(16, len(e.X0))) as p: res = p.map(work, range(len(e.X0)), chunksize=1)
    wall = time.perf_counter() - t0; Ts = [r['T_verify'] for r in res]; tq = np.array([r['t_query'] for r in res]); s = common.summarize(e, Ts)
    s.update(env=name, method='споры 6D коридор v6 NB%d NF%d K%d%s' % (NB, NF, K, ' wlim' if WL else ''), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(tq.sum()), t_wall_parallel=wall,
             hw='CPU 1 ядро/запрос, numpy+math-поток', per_start=res, Tref=e.Tref.tolist(), weight=common.deep_weight(back), nodes_back=int(len(back[0])))
    os.makedirs(common.DATA, exist_ok=True); json.dump(s, open(os.path.join(common.DATA, '%s_spores6d%s.json' % (name, tag)), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
