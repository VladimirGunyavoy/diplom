"""C5b: OCP на запрос — множественная стрельба (RK4, N интервалов), CasADi + IPOPT, свободное T (min T). Инициализация БЕЗ атласа:
линейная интерполяция состояний старт→цель, u малые случайные, T0 ∈ {3,6,10}; периодические углы — несколько обмоток цели (2πk, ближайшие к старту).
python ocp_casadi.py ENV [--N 60] [--procs 16]  (venv-ompl: casadi). Результат data/ENV_ocpc.json. Проверка — единый common.verify."""
import os, sys, time, json, argparse, itertools, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import casadi as ca
import common
ENV = None; NI = int(os.environ.get('OCPC_N', 60)); SUB = 4

def dyn(name, x, u):
    if name == 'pend': return ca.vertcat(x[1], ca.sin(x[0]) + u[0])
    if name == 'dd': return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])
    if name == 'di': return ca.vertcat(x[1], u[0])
    if name == 'di4': return ca.vertcat(x[2], x[3], u[0], u[1])
    if name in ('manip', 'mg', 'manipf'):
        A, B, D = common.Manip.AA, common.Manip.BB, common.Manip.DD
        c, s = ca.cos(x[1]), ca.sin(x[1]); w1, w2 = x[2], x[3]; h = -B * s
        r1 = u[0] - h * (2 * w1 * w2 + w2 ** 2); r2 = u[1] + h * w1 ** 2
        m11, m12, m22 = A + 2 * B * c, D + B * c, D; det = m11 * m22 - m12 ** 2
        return ca.vertcat(w1, w2, (m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det)
    if name == 'dp1':                                                                   # y = (q1 − π/2, q2, w1, w2); динамика r23 ref.acc / common.DP1
        G1, MS, S11, S12, S22 = common.DP1.G1, common.DP1.MS, common.DP1.S11, common.DP1.S12, common.DP1.S22
        th1 = x[0] + ca.pi / 2; th2 = th1 + x[1]; d1 = x[2]; d2 = x[2] + x[3]; c = ca.cos(th1 - th2); sn = ca.sin(th1 - th2)
        Q1 = u[0] - u[1] - G1 * MS[0] * ca.cos(th1) - S12 * sn * d2 ** 2; Q2 = u[1] - G1 * MS[1] * ca.cos(th2) + S12 * sn * d1 ** 2
        det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
        return ca.vertcat(x[2], x[3], a1, a2 - a1)
    raise ValueError(name)

def build(e, N):
    n, m = e.n, e.m; opti = ca.Opti(); T = opti.variable(); X = opti.variable(n, N + 1); U = opti.variable(m, N); x0 = opti.parameter(n); tg = opti.parameter(n)
    h = T / N / SUB
    def F(x, u):
        for _ in range(SUB):
            k1 = dyn(e.name, x, u); k2 = dyn(e.name, x + h / 2 * k1, u); k3 = dyn(e.name, x + h / 2 * k2, u); k4 = dyn(e.name, x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    opti.subject_to(X[:, 0] == x0)
    for k in range(N): opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k]))
    for i in range(m): opti.subject_to(opti.bounded(float(e.umin[i]), U[i, :], float(e.umax[i])))
    if e.name == 'dd': opti.subject_to(ca.fabs(U[0, :]) + ca.fabs(U[1, :]) <= 1)
    for i in range(n):
        if e.period[i] == 0: opti.subject_to(opti.bounded(float(e.lo[i]), X[i, :], float(e.hi[i])))
    for i in range(n): opti.subject_to(opti.bounded(-0.9 * float(e.half[i]), X[i, N] - tg[i], 0.9 * float(e.half[i])))
    if e.name == 'manipf':                                                              # запреты в узлах (+ запас .05): самостолкновение, стол
        for k in range(N + 1):
            q1, q2 = X[0, k], X[1, k]; ye = e.ML * ca.sin(q1); yt = ye + e.ML * ca.sin(q1 + q2)
            opti.subject_to(ca.pi - e.FD - ca.fabs(q2) >= .05); opti.subject_to(ye - e.TY >= .05); opti.subject_to(yt - e.TY >= .05)
    opti.subject_to(opti.bounded(.05, T, 40.)); opti.minimize(T)
    opti.solver('ipopt', dict(print_time=False), dict(print_level=0, max_iter=300, tol=1e-6, sb='yes'))
    return opti, T, X, U, x0, tg

def solve(e, y0, seed, N=NI, T0s=(3., 6., 10.)):
    rng = np.random.default_rng(seed); e.nf = 0; y0 = np.asarray(y0, float); opti, T, X, U, px0, ptg = build(e, N)
    per = [i for i in range(e.n) if e.period[i] > 0]; ks = []; g0 = np.asarray(getattr(e, 'goal', np.zeros(e.n)), float)
    for i in per:
        P = e.period[i]; ks.append([g0[i] + np.floor((y0[i] - g0[i]) / P) * P, g0[i] + np.ceil((y0[i] - g0[i]) / P) * P])      # две ближайшие обмотки цели (центр цели g0)
    best = None; nsolve = 0
    for comb in (itertools.product(*ks) if per else [()]):
        tg = g0.copy()
        for i, v in zip(per, comb): tg[i] = v
        for T0 in T0s:
            opti.set_value(px0, y0); opti.set_value(ptg, tg); a = np.linspace(0, 1, N + 1)
            opti.set_initial(X, y0[:, None] + (tg - y0)[:, None] * a[None]); opti.set_initial(U, rng.uniform(-.1, .1, (e.m, N)) * e.umax[:, None]); opti.set_initial(T, T0)
            nsolve += 1
            try: opti.solve()
            except RuntimeError: pass
            Tv = float(opti.debug.value(T)); Uv = np.asarray(opti.debug.value(U)).reshape(e.m, N)
            if not (np.isfinite(Tv) and Tv >= .05 - 1e-6 and np.isfinite(Uv).all()): continue      # невыполненные границы у необойдённого итерата
            Tc, why = common.verify(e, y0, Uv.T, np.full(N, Tv / N))
            if Tc is not None and (best is None or Tc < best): best = Tc
    return best, nsolve

def work(i):
    e = common.make(ENV); t0 = time.perf_counter(); b, ns = solve(e, e.X0[i], 100 + i, T0s=(1.5, 2.5, 4., 6., 10.) if ENV == 'manipf' else (3., 6., 10.)); return dict(i=i, T=b, t=time.perf_counter() - t0, nsolve=ns)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('env'); p.add_argument('--procs', type=int, default=16); p.add_argument('--only', type=int, default=None); a = p.parse_args(); ENV = a.env; e = common.make(a.env)
    if a.only is not None: print(work(a.only), 'Tref', e.Tref[a.only]); sys.exit()
    with Pool(a.procs) as pool: R = pool.map(work, range(len(e.X0)))
    s = common.summarize(e, [r['T'] for r in R]); ts = np.array([r['t'] for r in R])
    s.update(env=a.env, method='OCP CasADi+IPOPT мн.стрельба N=%d, свободный T, T0×обмотки (без атласа)' % NI, t_query_med=float(np.median(ts)), t_query_p90=float(np.percentile(ts, 90)), t_query_sum=float(ts.sum()), nf_med=float(np.median([r['nsolve'] for r in R])), hw='CPU 1 proc/запрос', per_start=R, Tref=e.Tref.tolist())
    json.dump(s, open(os.path.join(common.DATA, f'{a.env}_ocpc.json'), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
