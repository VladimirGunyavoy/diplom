"""линия A: OCP CasADi/IPOPT на запрос (multiple shooting N50 M4) на m3d. Варианты: multi — мультистарт 8 ветвей 2π × 4 затравки (= эталон, но без «знания»: перебор ветвей вслепую), fast — ближайшая ветвь × 2 затравки.
Время запроса — стена, 1 процесс (OMP 1); сборка символьной задачи (make_solver) — «заготовка». T — единым проверяльщиком (U кусочно-постоянно, N равных интервалов).
python ocp_6d.py m3d multi|fast [--procs 10] (venv-ompl: casadi)"""
import os, sys, time, json, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, common, envs_6d
ENVN = sys.argv[1]
if ENVN in ('m3gd', 'm3gu'): os.environ['G'] = '0.3'; os.environ['DOWN' if ENVN == 'm3gd' else 'UP'] = '1'
import importlib; L = importlib.import_module({'m3d': 'ocp_m3d_lib', 'm8': 'ocp_m8_lib'}.get(ENVN, 'ocp_m3g_lib'))
from multiprocessing import get_context
SOL = None; VAR = None
def work(i):
    e = common.make(ENVN); t0 = time.perf_counter()
    best, res = L.solve_query(e.X0[i], 50, 4, 4 if VAR == 'multi' else 2, seed=i, solver=SOL, nearest=(VAR == 'fast')); dt = time.perf_counter() - t0
    if best is None: return dict(i=i, T=None, t=dt, why='no_sol', nsolve=len(res))
    T, U, br = best; Tv, why = common.verify(e, e.X0[i], U, np.full(len(U), T / len(U))); return dict(i=i, T=Tv, T_ocp=float(T), t=dt, why=why, nsolve=len(res))
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('env'); p.add_argument('var'); p.add_argument('--procs', type=int, default=10); a = p.parse_args(); VAR = a.var; e = common.make(a.env)
    t0 = time.perf_counter(); SOL = L.make_solver(50, 4); t_build = time.perf_counter() - t0
    with get_context('fork').Pool(a.procs) as pool: R = pool.map(work, range(len(e.X0)), chunksize=1)
    s = common.summarize(e, [r['T'] for r in R]); ts = np.array([r['t'] for r in R])
    s.update(env=a.env, method='OCP CasADi/IPOPT multiple shooting N50 M4 (%s)' % ('мультистарт 8 ветвей×4 затравки' if a.var == 'multi' else 'ближайшая ветвь×2 затравки'), t_build=t_build, t_query_med=float(np.median(ts)), t_query_p90=float(np.percentile(ts, 90)), t_query_sum=float(ts.sum()),
             hw='CPU 1 процесс/запрос', per_start=R, Tref=e.Tref.tolist())
    os.makedirs(common.DATA, exist_ok=True); json.dump(s, open(os.path.join(common.DATA, '%s_ocp6d_%s.json' % (a.env, a.var)), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
