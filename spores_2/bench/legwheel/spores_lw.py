"""линия D: споры (коридор v6: обратное адаптивное дерево NB + прямое NF + refine IPOPT) на роботе пользователя. Масштабированные координаты (lw_dyn): окно цели |z_pos|<.3, |z_vel|<.6.
python spores_lw.py NB NF K [tag] — env: LWTU (.01), LWSC='sx,sψ,sα,sẋ,sψ̇,sα̇' (масштабы клеток), LWWB (предел |z_vel| в дереве, 'ok'), TRIES NEIGH RHO TAU; aida: venv-sb3 + PYTHONPATH=~/spore_v5/r5/pylib."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np, common
import env_lw as W, lw_dyn as L, lw_flow
sys.path.insert(0, os.path.join(HERE, '..', '..', 'v6'))
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
import multiprocessing as mp
NB, NF, K = [int(a) for a in sys.argv[1:4]]; tag = sys.argv[4] if len(sys.argv) > 4 else ''
e = W.LW(int(os.environ.get('LWN', 50))); tau = float(os.environ.get('TAU', .4)); rho = float(os.environ.get('RHO', .15)); Rq, Rw = L.RQ, L.RW
SC = np.array([float(v) for v in os.environ.get('LWSC', '2,3,1,10,40,8').split(',')]); WB = float(os.environ.get('LWWB', 400.))
LAY = L.LAYERS9 if os.environ.get('LW9') else L.LAYERS; fl = lw_flow.make_flow(LAY)
aok = (e.ALPHA_LO, e.ALPHA_HI)
ok = lambda p: (abs(p[3:]).max() <= WB) and (aok[0] < p[2] < aok[1])
ing = lambda P: (np.max(np.abs(P[..., :3]), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - x[0] ** 2, Rq ** 2 - x[1] ** 2, Rq ** 2 - x[2] ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(X[..., :3]), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(v * np.array([Rq] * 3 + [Rw] * 3))
S = SysN(fl, len(LAY), SC, ing, seeds, per=np.zeros(6), ok=ok)
import refine_lw; CAR = refine_lw.install(LAY, ms=int(os.environ.get('CAMS', 10)), maxit=int(os.environ.get('CAIT', 60)), prebuild=range(1, int(os.environ.get('CAPRE', 10))))
import tree_fast; tree_fast.install(); S.flow_all = lw_flow.make_flow_all(LAY, steps=int(os.environ.get('LWSTEPS', 8)))
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с, узлов %d' % (NB, t_build, len(back[0])), flush=True)
def work(i):
    t1 = time.perf_counter(); b = corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=int(os.environ.get('TRIES', 3)), kn=8, neigh=int(os.environ.get('NEIGH', 0)), nbest=1); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path')
    us = [LAY[s] for s in b[1]]; dts = list(b[2]); Tv, why = W.verify_real(e.X0[i], us, dts); return dict(i=i, T_agent=float(b[0]) * L.TU, t_query=dt, T_verify=Tv, why=why, nseg=len(us), us=us, dts=dts)
if __name__ == '__main__':
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 12)), len(e.X0))) as p: res = p.map(work, range(len(e.X0)), chunksize=1)
    wall = time.perf_counter() - t0; Ts = [r['T_verify'] for r in res]; tq = np.array([r['t_query'] for r in res]); s = common.summarize(e, Ts)
    s.update(env='lw', method='споры 6D коридор v6 NB%d NF%d K%d TU%g' % (NB, NF, K, L.TU), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(tq.sum()), t_wall_parallel=wall,
             hw='CPU 1 ядро/запрос, casadi-поток (gcc)', per_start=res, Tref=e.Tref.tolist(), weight=common.deep_weight(back), nodes_back=int(len(back[0])), reached_any=int(sum(t is not None for t in Ts)))
    json.dump(s, open(os.path.join(common.DATA, 'lw_spores%s.json' % tag), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
