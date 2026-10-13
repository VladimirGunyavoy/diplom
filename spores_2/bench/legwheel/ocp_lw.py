"""линия D: OCP-эталон на роботе (CasADi/IPOPT, multiple shooting N узлов, RK4 M шагов/интервал, свободное T, равные интервалы).
Ограничения: |u|≤1, окно цели (с запасом MG), α внутри механического предела, |x|≤BOUND_X; скорости НЕ ограничены (только физика робота).
python ocp_lw.py ref|fast [--procs 10] [--n 50]  → data/lw_ocp_{ref,fast}.json (+ lw_ref.json при ref: T проверяльщика)"""
import os, sys, time, json, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, casadi as ca
import env_lw as W, lw_dyn as L, common
MG = float(os.environ.get('LWMG', .02))                     # запас внутрь окна (доля .3/.6) — проверка в verify_real отсечёт остаток
def make_solver(N=50, M=4, bound_w=None):
    fx = ca.Function('f', [ca.SX.sym('x', 6), ca.SX.sym('u', 2)], [0]); x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 2); fx = ca.Function('f', [x, u], [L.f_ca(x, u)])
    x0 = ca.SX.sym('x0', 6); T = ca.SX.sym('T'); X = ca.SX.sym('X', 6, N); U = ca.SX.sym('U', 2, N); h = T / N / M; g = []; z = x0
    for k in range(N):
        for _ in range(M):
            k1 = fx(z, U[:, k]); k2 = fx(z + h / 2 * k1, U[:, k]); k3 = fx(z + h / 2 * k2, U[:, k]); k4 = fx(z + h * k3, U[:, k]); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k < N - 1: g.append(z - X[:, k]); z = X[:, k]
    gt = z; var = ca.vertcat(T, ca.vec(U), ca.vec(X[:, :N - 1]))
    G = ca.vertcat(*g, gt); nb = 6 * (N - 1)
    S = ca.nlpsol('S', 'ipopt', dict(x=var, f=T, g=G, p=x0), {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300, 'ipopt.tol': 1e-7, 'ipopt.acceptable_tol': 1e-5, 'ipopt.sb': 'yes'})
    return S, N
def bounds(N, bw):
    lbx = np.r_[1e-3, -np.ones(2 * N)]; ubx = np.r_[10., np.ones(2 * N)]
    xl = np.r_[-C_BX, -50, W.LW.ALPHA_LO, -bw * np.ones(3)] if False else None
    lo = np.r_[-1e3, -1e3, W.LW.ALPHA_LO, -bw, -bw, -bw]; hi = np.r_[1e3, 1e3, W.LW.ALPHA_HI, bw, bw, bw]
    lbx = np.r_[lbx, np.tile(lo, N - 1)]; ubx = np.r_[ubx, np.tile(hi, N - 1)]
    w = np.r_[np.full(3, L.RQ - MG * L.RQ), np.full(3, L.RW - MG * L.RW)]
    return lbx, ubx, w
def solve_query(x0, S, N, nseed, rng, bw=1e3, t_guess=(.4, .8, 1.6)):
    lbx, ubx, w = bounds(N, bw); best = None; res = []
    for s in range(nseed):
        T0 = t_guess[s % len(t_guess)]; U0 = rng.uniform(-1, 1, (2, N)) if s else np.sign(rng.uniform(-1, 1, (2, N))); X0 = np.linspace(x0, 0, N + 1)[1:-1].T * 0 + np.linspace(x0, np.zeros(6), N + 1)[1:N].T
        v0 = np.r_[T0, U0.ravel(order='F'), X0.ravel(order='F')]
        lbg = np.r_[np.zeros(6 * (N - 1)), -w]; ubg = np.r_[np.zeros(6 * (N - 1)), w]
        r = S(x0=v0, p=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']; ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'); v = np.array(r['x']).ravel(); res.append((float(v[0]), ok))
        if ok and (best is None or v[0] < best[0]): best = (float(v[0]), v[1:1 + 2 * N].reshape(2, N, order='F').T.copy())
    return best, res
def work(i):
    t0 = time.perf_counter(); e = ENV; best, res = solve_query(e.X0[i], SOL, N, NSEED, np.random.default_rng(i), bw=BW); dt = time.perf_counter() - t0
    if best is None: return dict(i=i, T=None, t=dt, why='no_sol', nsolve=len(res))
    T, U = best; Tv, why = W.verify_real(e.X0[i], U, np.full(N, T / N)); return dict(i=i, T=Tv, T_ocp=T * L.TU, t=dt, why=why, nsolve=len(res), U=U.tolist(), Tsc=T)
if __name__ == '__main__':
    from multiprocessing import get_context
    p = argparse.ArgumentParser(); p.add_argument('var'); p.add_argument('--procs', type=int, default=10); p.add_argument('--n', type=int, default=50); p.add_argument('--N', type=int, default=50); p.add_argument('--nseed', type=int, default=0); p.add_argument('--bw', type=float, default=1e3); a = p.parse_args()
    N = a.N; NSEED = a.nseed or (8 if a.var == 'ref' else 2); BW = a.bw; ENV = W.LW(a.n, ref_file='/nonexistent'); t0 = time.perf_counter(); SOL, _ = make_solver(N, 4); t_build = time.perf_counter() - t0
    with get_context('fork').Pool(a.procs) as pool: R = pool.map(work, range(len(ENV.X0)), chunksize=1)
    Tm = [r['T'] for r in R]; ts = np.array([r['t'] for r in R]); print('t_build %.1f с' % t_build, 'T мед', np.nanmedian([np.nan if t is None else t for t in Tm]), 'дошли', sum(t is not None for t in Tm), '/', len(R), 'запрос мед %.1f' % np.median(ts), flush=True)
    out = dict(env='lw', method='OCP CasADi/IPOPT multiple shooting N%d M4 (%s)' % (N, a.var), t_build=t_build, t_query_med=float(np.median(ts)), t_query_p90=float(np.percentile(ts, 90)), t_query_sum=float(ts.sum()), per_start=R, hw='CPU 1 процесс/запрос', nseed=NSEED)
    os.makedirs(common.DATA, exist_ok=True); json.dump(out, open(os.path.join(common.DATA, 'lw_ocp_%s.json' % a.var), 'w'), default=float)
    if a.var == 'ref': json.dump([r['T'] if r['T'] is not None else float('nan') for r in R], open(os.path.join(common.DATA, 'lw_ref.json'), 'w'))
