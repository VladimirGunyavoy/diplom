"""линия A: OCP CasADi/IPOPT для 4 звеньев 8D + g .3, цель вниз (обобщение ocp_m3g_lib.py на n=4): multiple shooting N50 M4, мультистарт по ветвям 2π (2^4) × затравки."""
import os, sys, time, itertools
import numpy as np
import casadi as ca
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manip3g_dyn import f as f_np, MS, L
n = 4; GG = .3; C3 = np.zeros(n); C3[0] = -np.pi / 2; Rq, Rw, WM = .3, .6, 3.
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
def f_ca(x, u, g=GG, m=MS, ln=L):
    q, w = x[:n], x[n:]; m = np.asarray(m[:n]); ln = np.asarray(ln[:n])
    th = [ca.sum1(q[:i + 1]) for i in range(n)]; thd = [ca.sum1(w[:i + 1]) for i in range(n)]
    S = [[float(m[max(i, j):].sum() * ln[i] * ln[j]) for j in range(n)] for i in range(n)]; M = ca.SX(n, n); rhs = ca.SX(n, 1)
    for i in range(n):
        for j in range(n): M[i, j] = S[i][j] * ca.cos(th[i] - th[j])
        rhs[i] = u[i] - (u[i + 1] if i + 1 < n else 0) - g * ln[i] * float(m[i:].sum()) * ca.cos(th[i]) - sum(S[i][j] * ca.sin(th[i] - th[j]) * thd[j] ** 2 for j in range(n))
    tdd = ca.solve(M, rhs); return ca.vertcat(w, tdd[0], *[tdd[i] - tdd[i - 1] for i in range(1, n)])
def make_solver(N, M):
    nx_ = 2 * n; x = ca.SX.sym('x', nx_); u = ca.SX.sym('u', n); h = ca.SX.sym('h'); fx = ca.Function('f', [x, u], [f_ca(x, u)]); z = x
    for _ in range(M):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F = ca.Function('F', [x, u, h], [z]); X = ca.SX.sym('X', nx_, N + 1); U = ca.SX.sym('U', n, N); T = ca.SX.sym('T'); P = ca.SX.sym('P', nx_ + n)
    h_ = T / (N * M); g = [X[:, 0] - P[:nx_]]
    for k in range(N): g.append(F(X[:, k], U[:, k], h_) - X[:, k + 1])
    g.append(X[:n, N] - P[nx_:]); w = ca.vertcat(ca.vec(X), ca.vec(U), T)
    S = ca.nlpsol('S', 'ipopt', {'x': w, 'f': T, 'g': ca.vertcat(*g), 'p': P}, {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1500, 'ipopt.tol': 1e-8, 'ipopt.constr_viol_tol': 1e-9})
    nx = nx_ * (N + 1); nu = n * N; lbx = np.full(nx + nu + 1, -np.inf); ubx = np.full(nx + nu + 1, np.inf); Xi = np.arange(nx).reshape(N + 1, nx_)
    lbx[Xi[:, n:].ravel()] = -WM; ubx[Xi[:, n:].ravel()] = WM; lbx[Xi[-1, n:]] = -Rw; ubx[Xi[-1, n:]] = Rw; lbx[nx:nx + nu] = -1; ubx[nx:nx + nu] = 1; lbx[-1] = .05; ubx[-1] = 14
    ng = nx_ * (N + 1) + n; lbg = np.zeros(ng); ubg = np.zeros(ng); lbg[-n:] = -Rq; ubg[-n:] = Rq
    return S, lbx, ubx, lbg, ubg
def simulate(x0, U, T, dt=.005):
    N = U.shape[0]; x = np.array(x0, float); hseg = T / N; k = max(1, int(np.ceil(hseg / dt))); h = hseg / k; wmax = np.max(np.abs(x[n:]))
    for i in range(N):
        for _ in range(k):
            k1 = f_np(x, U[i], GG); k2 = f_np(x + h / 2 * k1, U[i], GG); k3 = f_np(x + h / 2 * k2, U[i], GG); k4 = f_np(x + h * k3, U[i], GG); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        wmax = max(wmax, np.max(np.abs(x[n:])))
    return x, wmax
def solve_query(x0, N=50, M=4, nstart=4, seed=0, solver=None, nearest=False):
    S, lbx, ubx, lbg, ubg = solver or make_solver(N, M); nx_ = 2 * n
    d0 = wr(C3 - x0[:n]); rng = np.random.default_rng(seed); best = None; res = []
    for br in ([tuple(d0)] if nearest else itertools.product(*[(d, d - 2 * np.pi * np.sign(d)) for d in d0])):
        tgt = x0[:n] + np.array(br); P = np.concatenate([x0, tgt])
        for s in range(nstart):
            T0 = [2., 3., 4., 6.][s % 4] * (1 + .1 * rng.standard_normal()); al = np.linspace(0, 1, N + 1)[:, None]; Xg = (1 - al) * x0 + al * np.concatenate([tgt, np.zeros(n)])
            Xg[:, n:] = np.clip(Xg[:, n:] + (np.array(br) / T0)[None] * np.sin(np.pi * al) * 1.5, -WM, WM); Ug = np.clip(.5 * rng.standard_normal((N, n)), -1, 1)
            t0 = time.time(); r = S(x0=np.concatenate([Xg.ravel(), Ug.ravel(), [T0]]), p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
            wv = np.array(r['x']).ravel(); T = wv[-1]; U = wv[nx_ * (N + 1):nx_ * (N + 1) + n * N].reshape(N, n); xe, wmax = simulate(x0, U, T)
            ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(wr(xe[:n] - C3)) < Rq + 1e-3) and np.all(np.abs(xe[n:]) < Rw + 1e-3) and wmax <= WM + 1e-2
            res.append((ok, T, br, s, st, time.time() - t0))
            if ok and (best is None or T < best[0]): best = (T, U, br)
    return best, res
