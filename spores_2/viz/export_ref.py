"""viz (линия B): эталон-«призрак» — OCP минимального времени (CasADi + IPOPT, множественная стрельба, как bench/ocp_casadi.py, плюс диски-препятствия в узлах) → эпизод
«эталон (OCP)» в viz/data/<система>.json на тех же стартах, что у спор. Запуск на aida (venv-ompl): ~/venv-ompl/bin/python viz/export_ref.py dd|ddobst|car|carfree [--procs 16]
T эталона пересчитан общим проверяльщиком (common.verify); T_ref в эпизоде — тот же, что у спор (эталон бенча: dd_refbox / car_ref20), так что T/T_ref эталона — честные."""
import os, sys, json, itertools, argparse, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'bench')); sys.path.insert(0, HERE)
import casadi as ca, common, export_mobile as EM
from multiprocessing import Pool
ENVN = {'dd': 'dd', 'ddobst': 'ddobst7', 'car': 'car', 'carfree': 'car'}
def make(sysname):
    e = common.make(ENVN[sysname])
    if sysname == 'carfree': e.HZ = np.array([[1e3, 1e3]])
    return e
def discs(e):
    if e.name.startswith('ddobst'): return [(c[0], c[1], c[2]) for c in e.DISKS]
    if e.name == 'car': return [(h[0], h[1], e.HR) for h in e.HZ]
    return []
def dyn(e, x, u):
    if e.name.startswith('dd'): return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])
    return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), u[1], u[0] - e.K * x[3])
def solve(args):
    sysname, i, N = args; e = make(sysname); y0 = np.asarray(e.X0[i], float); n, m = e.n, e.m; DS = discs(e); SUB = 2; best = None
    car = e.name == 'car'; g0 = np.zeros(n) if not car else np.array([e.GOAL[0], e.GOAL[1], 0., 0.])
    per = [k for k in range(n) if e.period[k] > 0 and (not car or k == 2)]; ks = []
    for k in per: P = e.period[k]; ks.append([g0[k] + np.floor((y0[k] - g0[k]) / P) * P, g0[k] + np.ceil((y0[k] - g0[k]) / P) * P])
    opti = ca.Opti(); T = opti.variable(); X = opti.variable(n, N + 1); U = opti.variable(m, N); px0 = opti.parameter(n); ptg = opti.parameter(n); h = T / N / SUB
    def F(x, u):
        for _ in range(SUB): k1 = dyn(e, x, u); k2 = dyn(e, x + h / 2 * k1, u); k3 = dyn(e, x + h / 2 * k2, u); k4 = dyn(e, x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    opti.subject_to(X[:, 0] == px0)
    for k in range(N): opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k]))
    for j in range(m): opti.subject_to(opti.bounded(float(e.umin[j]), U[j, :], float(e.umax[j])))
    if e.name.startswith('dd'): opti.subject_to(ca.fabs(U[0, :]) + ca.fabs(U[1, :]) <= 1)
    for j in range(n):
        if e.period[j] == 0: opti.subject_to(opti.bounded(float(e.lo[j]), X[j, :], float(e.hi[j])))
    if car: opti.subject_to((X[0, N] - e.GOAL[0]) ** 2 + (X[1, N] - e.GOAL[1]) ** 2 <= (.9 * e.GR) ** 2)
    else:
        for j in range(n): opti.subject_to(opti.bounded(-.9 * float(e.half[j]), X[j, N] - ptg[j], .9 * float(e.half[j])))
    for (cx, cy, r) in DS:
        for k in range(N + 1): opti.subject_to((X[0, k] - cx) ** 2 + (X[1, k] - cy) ** 2 >= (r + .03) ** 2)
    opti.subject_to(opti.bounded(.05, T, 40.)); opti.minimize(T); opti.solver('ipopt', dict(print_time=False), dict(print_level=0, max_iter=400, tol=1e-6, sb='yes'))
    rng = np.random.default_rng(100 + i); a = np.linspace(0, 1, N + 1)
    for comb in (itertools.product(*ks) if per else [()]):
        tg = g0.copy()
        for k, v in zip(per, comb): tg[k] = v
        tgt = tg.copy()
        if car: tgt[2] = np.arctan2(e.GOAL[1] - y0[1], e.GOAL[0] - y0[0])
        for T0 in (3., 5., 8.):
            for jit in (0, 1):                                                       # jit — боковой прогиб начального пути (обход дисков)
                z0 = y0[:, None] + (tgt - y0)[:, None] * a[None]
                if jit and DS: d = (tgt - y0)[:2]; nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9); z0[:2] += (jit * 1.2 * np.sin(np.pi * a))[None] * nrm[:, None]
                opti.set_value(px0, y0); opti.set_value(ptg, tg); opti.set_initial(X, z0); opti.set_initial(U, rng.uniform(-.1, .1, (m, N)) * e.umax[:, None]); opti.set_initial(T, T0)
                try: opti.solve()
                except RuntimeError: pass
                Tv = float(opti.debug.value(T)); Uv = np.asarray(opti.debug.value(U)).reshape(m, N)
                if not (np.isfinite(Tv) and np.isfinite(Uv).all()): continue
                Tc, why = common.verify(e, y0, Uv.T, np.full(N, Tv / N))
                if Tc is not None and (best is None or Tc < best[0]): best = (Tc, Uv.copy(), Tv)
    return sysname, i, best
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('system'); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--N', type=int, default=70); a = ap.parse_args()
    p = os.path.join(HERE, 'data', a.system + '.json'); d = json.load(open(p)); e = make(a.system); starts = sorted({ep['start'] for ep in d['episodes']})
    with Pool(min(a.procs, len(starts))) as pool: R = pool.map(solve, [(a.system, i, a.N) for i in starts])
    d['episodes'] = [ep for ep in d['episodes'] if 'OCP' not in ep['method']]; src = {'dd': 'dd_ocpc.json', 'ddobst': 'нет в бенче (OCP CasADi этой выгрузки)', 'car': 'нет в бенче (OCP CasADi этой выгрузки; эталон бенча car_ref20 — только T)', 'carfree': 'OCP CasADi этой выгрузки'}[a.system]
    for _, i, best in R:
        tref = next((ep['T_ref'] for ep in d['episodes'] if ep['start'] == i), float('nan'))
        if best is None: print(a.system, i, 'OCP не сошёлся', flush=True); continue
        Tc, Uv, Tv = best; segs = [(Uv[:, k], Tv / Uv.shape[1]) for k in range(Uv.shape[1])]; ep = EM.episode(e, i, 'эталон (OCP)', segs, tref, src); d['episodes'].append(ep); print(a.system, i, 'T', ep['T'], 'T_ref', ep['T_ref'], flush=True)
    if a.system == 'carfree':                                                          # эталона бенча нет: за T_ref берём оптимум OCP этой же постановки
        To = {ep['start']: ep['T'] for ep in d['episodes'] if 'OCP' in ep['method'] and ep['T']}
        for ep in d['episodes']: ep['T_ref'] = round(To.get(ep['start'], ep['T_ref']), 3)
        d['title'] = 'Машина без препятствий (эталон — OCP CasADi этой выгрузки: в бенче для этой постановки эталона нет)'
    EM.finalize(d); json.dump(d, open(p, 'w'), ensure_ascii=False, separators=(',', ':')); print(p, os.path.getsize(p) // 1024, 'КБ')
