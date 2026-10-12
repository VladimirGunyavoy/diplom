"""RACE: эталон OCP (CasADi + IPOPT) — минимальное время до финиша по трассе, множественная стрельба RK4, свободное T, gfree в узлах (sdf — B-сплайн интерполянт);
старт инициализации — осевая линия трассы (скорость v0..v_init), мультистарт по v_init. Эталон вне таймера (как C5). python race_ocp.py ref [--procs 16] → data/race_ref.json; python race_ocp.py one I.
Времена — для таблицы «OCP на запрос» (t за старт, число решений)."""
import os, sys, time, json, argparse, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casadi as ca, scipy.ndimage as nd
import common, race as R
NI = int(os.environ.get('RACE_N', 200)); SUB = 2; MG = R.MARGIN + .15
_S = nd.gaussian_filter(R.SDF[::2, ::2] * R.PXM, .7); _gy = np.arange(_S.shape[0]) * 2 * R.PXM; _gx = np.arange(_S.shape[1]) * 2 * R.PXM
SDFI = ca.interpolant('sdf', 'bspline', [_gy, _gx], _S.ravel(order='F'))          # порядок: sdf(y, x), column-major

def dyn(x, u):
    th, v, k = x[2], x[3], x[4]; ad = ca.sqrt(ca.fmax(R.MUG ** 2 - (v * v * k) ** 2, 1e-6)); sa = u[1]
    a = ca.if_else(sa > 0, sa * ca.fmin(R.AMAX * ca.fmax(1 - v / R.VMAX, 0), ad), sa * ca.fmin(R.BMAX, ad))
    return ca.vertcat(v * ca.cos(th), v * ca.sin(th), v * k, a, R.KRATE * u[0])

def build(N):
    opti = ca.Opti(); T = opti.variable(); X = opti.variable(5, N + 1); U = opti.variable(2, N); x0 = opti.parameter(5); h = T / N / SUB
    def F(x, u):
        for _ in range(SUB):
            k1 = dyn(x, u); k2 = dyn(x + h / 2 * k1, u); k3 = dyn(x + h / 2 * k2, u); k4 = dyn(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    opti.subject_to(X[:, 0] == x0)
    for k in range(N): opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k]))
    opti.subject_to(opti.bounded(-1, U, 1)); opti.subject_to(opti.bounded(0, X[3, :], R.VMAX)); opti.subject_to(opti.bounded(-R.KMAX, X[4, :], R.KMAX))
    opti.subject_to(X[3, :] ** 2 * ca.fabs(X[4, :]) <= R.MUG)
    for k in range(N + 1): opti.subject_to(SDFI(ca.vertcat(X[1, k], X[0, k])) >= MG)
    tmid = R.FIN_Y + R.FIN_W / 2; opti.subject_to(X[1, N] == tmid); opti.subject_to(opti.bounded(R.FIN_X[0] + .5, X[0, N], R.FIN_X[1] - .5)); opti.subject_to(T >= .5)
    opti.minimize(T + 1e-4 * ca.sumsqr(U[0, :]) / N)
    opti.solver('ipopt', dict(print_time=False), dict(print_level=0, max_iter=500, tol=1e-6, sb='yes'))
    return opti, T, X, U, x0

def init_path(y0, N, vi, T0):
    """начальное приближение: по оси от старта до финиша (+ круг, если стартовый s после финиша), скорость vi, θ без обёрток"""
    s0 = float(R.progress(y0[None])[0]); L = R.LAP - s0 if s0 > 1 else 0.; ss = s0 + np.linspace(0, max(vi * T0, L), N + 1) if False else s0 + np.linspace(0, R.LAP - s0, N + 1)
    xy, th = R.cl_at(ss); th = np.unwrap(th); th += 2 * np.pi * np.round((y0[2] - th[0]) / (2 * np.pi)); return np.c_[xy, th, np.full(N + 1, vi), np.zeros(N + 1)].T

def solve(e, y0, seed=0, N=NI):
    y0 = np.asarray(y0, float); opti, T, X, U, px0 = build(N); best = None; ns = 0; x_bst = None
    for vi in (6., 10.):
        Xi = init_path(y0, N, vi, 0); Xi[:, 0] = y0
        # θ в конце — вниз (sin > 0): корректировка обмотки под границы финиша
        opti.set_value(px0, y0); opti.set_initial(X, Xi); opti.set_initial(U, 0.); opti.set_initial(T, (R.LAP - float(R.progress(y0[None])[0])) / vi)
        ns += 1
        try: opti.solve()
        except RuntimeError: pass
        Tv = float(opti.debug.value(T)); Uv = np.asarray(opti.debug.value(U)).reshape(2, N)
        if not (np.isfinite(Tv) and np.isfinite(Uv).all()): continue
        Tc, why = common.verify(e, y0, Uv.T, np.full(N, Tv / N))
        if Tc is not None and (best is None or Tc < best): best = Tc
    return best, ns

def work(i):
    e = R.Race(); t0 = time.perf_counter(); b, ns = solve(e, e.X0[i], i); return dict(i=i, Tref=b, t=time.perf_counter() - t0, nsolve=ns, x0=e.X0[i].tolist())

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd'); p.add_argument('i', nargs='?', type=int); p.add_argument('--procs', type=int, default=16); a = p.parse_args()
    if a.cmd == 'one': print(work(a.i))
    else:
        from multiprocessing import Pool
        with Pool(a.procs) as pool: RR = pool.map(work, range(len(R.Race().X0)))
        json.dump(RR, open(os.path.join(common.DATA, 'race_ref.json'), 'w'), indent=1, default=float); print(sum(r['Tref'] is not None for r in RR), '/', len(RR), 'эталонов')
