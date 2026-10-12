"""C5: OCP на запрос — прямая стрельба (кусочно-постоянное u, N интервалов), SLSQP (scipy), min T при goal_dist ≤ 0, мультистарт без знания числа переключений: начальные u — случайные малые, T0 ∈ {2,4,8}.
python ocp_run.py ENV [--N 40] [--procs 16]. Время на запрос, T/T_ref, доля. Метод и инициализация слабее эталона нарочно (эталон знает структуру)."""
import os, sys, time, json, argparse, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from scipy.optimize import minimize
import common
ENV = None; NI = 40
def solve(e, y0, seed, N=NI, T0s=(2., 4., 8.), maxit=200):
    rng = np.random.default_rng(seed); best = None; m = e.m; e.nf = 0
    def roll(z):
        T = z[0]; U = z[1:].reshape(N, m); y = np.asarray(y0, float)[None]; h = T / N
        for k in range(N): y = e.step(y, U[k:k + 1], h)
        return y
    cons = [dict(type='ineq', fun=lambda z: -float(e.goal_dist(roll(z))[0]))]
    for T0 in T0s:
        z0 = np.concatenate([[T0], rng.uniform(-.3, .3, N * m)]); bnds = [(.05, 40.)] + [(float(e.umin[k % m]), float(e.umax[k % m])) for k in range(N * m)]
        r = minimize(lambda z: z[0], z0, jac=lambda z: np.eye(1, len(z), 0)[0], bounds=bnds, constraints=cons, method='SLSQP', options=dict(maxiter=maxit, ftol=1e-9))
        if r.success or True:
            T, why = common.verify(e, y0, r.x[1:].reshape(N, m), np.full(N, r.x[0] / N))
            if T is not None and (best is None or T < best[0]): best = (T, r.x.copy())
    return best, int(e.nf)
def work(i):
    e = common.make(ENV); t0 = time.perf_counter(); b, nf = solve(e, e.X0[i], 100 + i); t = time.perf_counter() - t0
    return dict(i=i, T=None if b is None else float(b[0]), t=t, nf=nf)
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('env'); p.add_argument('--procs', type=int, default=16); a = p.parse_args(); ENV = a.env; e = common.make(a.env)
    with Pool(a.procs) as pool: R = pool.map(work, range(len(e.X0)))
    s = common.summarize(e, [r['T'] for r in R]); ts = np.array([r['t'] for r in R]); s.update(env=a.env, method='OCP SLSQP shooting N=%d multistart 3' % NI, t_query_med=float(np.median(ts)), t_query_p90=float(np.percentile(ts, 90)), t_query_sum=float(ts.sum()), nf_med=float(np.median([r['nf'] for r in R])), hw='CPU 1 proc/запрос', per_start=R, Tref=e.Tref.tolist())
    json.dump(s, open(os.path.join(common.DATA, f'{a.env}_ocp.json'), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
