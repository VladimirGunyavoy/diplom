"""линия D: ДИСКРЕТНЫЙ эталон на роботе — управление кусочно-постоянное на сетке control_dt = PHYSICS_DT = .002 с (как у PPO), RK4 M=1 на шаг (интегратор робота).
Один решатель на NMAX шагов; число активных шагов N задаётся маской (состояние после N-го шага замораживается), цель — в окне на шаге N. Минимальное N ищется перебором от нижней
оценки (непрерывный эталон lw_ocp_ref.json) вверх. T = N·.002 с. Пределы: |u|≤1, α-предел в узлах, окно цели с запасом MG; проверка — verify_real (тот же rk4_step).
python ocp_lw_d.py [--n 50] [--procs 12]  → data/lw_ocp_d.json, lw_ref.json (T проверяльщика)"""
import os, sys, time, json, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, casadi as ca
import env_lw as W, lw_dyn as L, common
from lw_dyn import C
MG = float(os.environ.get('LWMG', .02)); NMAX = int(os.environ.get('LWNMAX', 120)); H = C.PHYSICS_DT / L.TU
def make_solver():
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 2); fx = ca.Function('f', [x, u], [L.f_ca(x, u)])
    x0 = ca.SX.sym('x0', 6); msk = ca.SX.sym('m', NMAX); X = ca.SX.sym('X', 6, NMAX); U = ca.SX.sym('U', 2, NMAX); z = x0; g = []
    for k in range(NMAX):
        k1 = fx(z, U[:, k]); k2 = fx(z + H / 2 * k1, U[:, k]); k3 = fx(z + H / 2 * k2, U[:, k]); k4 = fx(z + H * k3, U[:, k]); zn = z + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        zn = z + msk[k] * (zn - z)
        g.append(zn - X[:, k]); z = X[:, k]
    var = ca.vertcat(ca.vec(U), ca.vec(X)); G = ca.vertcat(*g)
    S = ca.nlpsol('S', 'ipopt', dict(x=var, f=1e-3 * ca.sumsqr(U), g=G, p=ca.vertcat(x0, msk)), {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 400, 'ipopt.tol': 1e-7, 'ipopt.acceptable_tol': 1e-5, 'ipopt.sb': 'yes'})
    return S
def solve_N(S, x0, N, rng, seeds=3, Uinit=None):
    msk = np.r_[np.ones(N), np.zeros(NMAX - N)]; w = np.r_[np.full(3, L.RQ), np.full(3, L.RW)] * (1 - MG)
    lo = np.r_[-1e3, -1e3, W.LW.ALPHA_LO, -1e3, -1e3, -1e3]; hi = np.r_[1e3, 1e3, W.LW.ALPHA_HI, 1e3, 1e3, 1e3]
    lbx = np.r_[-np.ones(2 * NMAX), np.tile(lo, NMAX)]; ubx = -lbx * 0 + np.r_[np.ones(2 * NMAX), np.tile(hi, NMAX)]
    # терминал: узел N-1 (индекс) в окне; остальные узлы свободны; узлы после N совпадают с ним по маске
    lbx = lbx.copy(); ubx = ubx.copy(); off = 2 * NMAX + 6 * (N - 1)
    # окно через границы переменных последнего активного узла; α-предел ужесточаем окном (оно внутри предела)
    lbx[off:off + 6] = -w; ubx[off:off + 6] = w
    for s in range(seeds):
        U0 = (Uinit if (Uinit is not None and s == 0) else rng.uniform(-1, 1, (2, NMAX))); X0 = np.tile(np.linspace(0, 1, NMAX)[None], (6, 1)) * 0 + np.linspace(x0, np.zeros(6), NMAX).T
        r = S(x0=np.r_[U0.ravel(order='F'), X0.ravel(order='F')], p=np.r_[x0, msk], lbx=lbx, ubx=ubx, lbg=0, ubg=0); st = S.stats()['return_status']
        if st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): return np.array(r['x']).ravel()[:2 * NMAX].reshape(2, NMAX, order='F').T[:N].copy()
    return None
def work(i):
    t0 = time.perf_counter(); e = ENV; x0 = e.X0[i]; rng = np.random.default_rng(i); Tc = CONT[i]; n0 = max(1, int(Tc / C.PHYSICS_DT * L.TU)) if np.isfinite(Tc) else 2
    n0 = max(1, int(Tc / C.PHYSICS_DT)) if np.isfinite(Tc) else 2          # Tc — в секундах
    cand = list(range(n0, min(NMAX, n0 + 25) + 1)) if np.isfinite(Tc) else [3, 5, 8, 12, 16, 22, 30, 40, 55, 75, 100, 120]
    if not np.isfinite(Tc): cand = list(range(2, 13)) + cand[3:]          # без оценки: все N от 2, потом редкая сетка; ниже, после попадания на грубой сетке, добираем меньшие N подряд
    for N in cand:
        U = solve_N(SOL, x0, N, rng)
        if U is None: continue
        Tv, why = W.verify_real(x0, U, np.full(N, H))
        if why == 'ok' and not np.isfinite(Tc) and N > 12:
            for N2 in range(cand[cand.index(N) - 1] + 1, N):
                U2 = solve_N(SOL, x0, N2, rng)
                if U2 is not None:
                    T2, w2 = W.verify_real(x0, U2, np.full(N2, H))
                    if w2 == 'ok': N, U, Tv = N2, U2, T2; break
        if why == 'ok': return dict(i=i, T=Tv, N=N, t=time.perf_counter() - t0, why=why, U=U.tolist(), tried=len(cand))
    return dict(i=i, T=None, t=time.perf_counter() - t0, why='no_sol')
if __name__ == '__main__':
    from multiprocessing import get_context
    p = argparse.ArgumentParser(); p.add_argument('--n', type=int, default=50); p.add_argument('--procs', type=int, default=12); a = p.parse_args()
    ENV = W.LW(a.n, ref_file='/nonexistent'); cf = os.path.join(common.DATA, 'lw_ocp_ref.json')
    CONT = np.full(a.n, np.nan)
    if os.path.exists(cf):
        for r in json.load(open(cf))['per_start'][:a.n]: CONT[r['i']] = r['T'] if r['T'] is not None else np.nan
    t0 = time.perf_counter(); SOL = make_solver(); t_build = time.perf_counter() - t0; print('сборка', t_build, flush=True)
    with get_context('fork').Pool(a.procs) as pool: R = pool.map(work, range(a.n), chunksize=1)
    Tm = [r['T'] for r in R]; ts = np.array([r['t'] for r in R]); print('T мед', np.nanmedian([np.nan if t is None else t for t in Tm]), 'дошли', sum(t is not None for t in Tm), '/', len(R), 'запрос мед %.1f' % np.median(ts), flush=True)
    out = dict(env='lw', method='OCP дискретный (сетка .002 с, RK4 M=1), мин. N', t_build=t_build, t_query_med=float(np.median(ts)), t_query_p90=float(np.percentile(ts, 90)), per_start=R, hw='CPU 1 процесс/запрос')
    json.dump(out, open(os.path.join(common.DATA, 'lw_ocp_d.json'), 'w'), default=float)
    json.dump([r['T'] if r['T'] is not None else float('nan') for r in R], open(os.path.join(common.DATA, 'lw_ref.json'), 'w'))
