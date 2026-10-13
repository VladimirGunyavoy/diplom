"""CA4-6D+2 диска (w28): OCP CasADi/IPOPT для 3 звеньев 6D + g .3 с произвольной целью (B.C3) и двумя дисками (зазор звеньев ≥ CLR0 в узлах multiple shooting; итог проверяется мелко numpy: зазор ≥ 0 вдоль пути).
Диски — те же, что в v6 stats_manip6 OBST (all_systems 8d): ((1,-1.5), .3), ((-1,-1), .3)."""
import os, sys, time, itertools
os.environ.setdefault('G', '0.3'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, casadi as ca
import ocp_m3g_lib as B
from manip3g_dyn import clearance, L as LN
OB = [((1.0, -1.5), 0.3), ((-1.0, -1.0), 0.3)]; CLR0 = .02; wr = B.wr; Rq, Rw, WM = B.Rq, B.Rw, B.WM
def clr_ca(q):
    th = [ca.sum1(q[:i + 1]) for i in range(3)]; pts = [ca.DM([0, 0])]
    for i in range(3): pts.append(pts[-1] + float(LN[i]) * ca.vertcat(ca.cos(th[i]), ca.sin(th[i])))
    out = []
    for c, r in OB:
        c = ca.DM(c)
        for i in range(3):
            a, b = pts[i], pts[i + 1]; d = b - a; u = ca.fmin(ca.fmax(ca.dot(c - a, d) / ca.fmax(ca.dot(d, d), 1e-12), 0), 1); out.append(ca.norm_2(a + u * d - c) - r)
    return ca.vertcat(*out)
def make_solver(N, M):
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 3); h = ca.SX.sym('h'); fx = ca.Function('f', [x, u], [B.f_ca(x, u)]); z = x
    for _ in range(M):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F = ca.Function('F', [x, u, h], [z]); qs = ca.SX.sym('qs', 3); CL = ca.Function('CL', [qs], [clr_ca(qs)])
    X = ca.SX.sym('X', 6, N + 1); U = ca.SX.sym('U', 3, N); T = ca.SX.sym('T'); P = ca.SX.sym('P', 12); h_ = T / (N * M); g = [X[:, 0] - P[:6]]
    for k in range(N): g.append(F(X[:, k], U[:, k], h_) - X[:, k + 1])
    g.append(X[:3, N] - P[6:9]); no = 6
    for k in range(N + 1): g.append(CL(X[:3, k]))
    w = ca.vertcat(ca.vec(X), ca.vec(U), T)
    S = ca.nlpsol('S', 'ipopt', {'x': w, 'f': T, 'g': ca.vertcat(*g), 'p': P}, {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1500, 'ipopt.tol': 1e-8, 'ipopt.constr_viol_tol': 1e-9})
    nx = 6 * (N + 1); nu = 3 * N; lbx = np.full(nx + nu + 1, -np.inf); ubx = np.full(nx + nu + 1, np.inf); Xi = np.arange(nx).reshape(N + 1, 6)
    lbx[Xi[:, 3:].ravel()] = -WM; ubx[Xi[:, 3:].ravel()] = WM; lbx[Xi[-1, 3:]] = -Rw; ubx[Xi[-1, 3:]] = Rw; lbx[nx:nx + nu] = -1; ubx[nx:nx + nu] = 1; lbx[-1] = 0.05; ubx[-1] = 12
    ng0 = 6 * (N + 1) + 3; ng = ng0 + no * (N + 1); lbg = np.zeros(ng); ubg = np.zeros(ng); lbg[ng0 - 3:ng0] = -Rq; ubg[ng0 - 3:ng0] = Rq; lbg[ng0:] = CLR0; ubg[ng0:] = np.inf
    return S, lbx, ubx, lbg, ubg
def simulate(x0, U, T, dt=0.005):
    N = U.shape[0]; x = np.array(x0, float); hseg = T / N; k = max(1, int(np.ceil(hseg / dt))); h = hseg / k; wmax = np.max(np.abs(x[3:])); mc = clearance(x[:3], OB)
    for i in range(N):
        for _ in range(k):
            k1 = B.f_np(x, U[i], B.GG); k2 = B.f_np(x + h / 2 * k1, U[i], B.GG); k3 = B.f_np(x + h / 2 * k2, U[i], B.GG); k4 = B.f_np(x + h * k3, U[i], B.GG); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            wmax = max(wmax, np.max(np.abs(x[3:]))); mc = min(mc, float(clearance(x[:3], OB)))
    return x, wmax, mc
def solve_query(x0, goal, N=50, M=4, nstart=4, seed=0, solver=None, nearest=False):
    S, lbx, ubx, lbg, ubg = solver or make_solver(N, M); C3 = np.asarray(goal, float); d0 = wr(C3 - x0[:3]); rng = np.random.default_rng(seed); best = None; res = []
    for br in ([tuple(d0)] if nearest else itertools.product(*[(d, d - 2 * np.pi * np.sign(d)) for d in d0])):
        tgt = x0[:3] + np.array(br); P = np.concatenate([x0, tgt, np.zeros(3)])
        for s in range(nstart):
            T0 = [1.5, 2.5, 3.5, 5.0][s % 4] * (1 + 0.1 * rng.standard_normal()); al = np.linspace(0, 1, N + 1)[:, None]; Xg = (1 - al) * x0 + al * np.concatenate([tgt, np.zeros(3)])
            Xg[:, 3:] = np.clip(Xg[:, 3:] + (np.array(br) / T0)[None] * np.sin(np.pi * al) * 1.5, -WM, WM); Ug = np.clip(0.5 * rng.standard_normal((N, 3)), -1, 1)
            t0 = time.time(); r = S(x0=np.concatenate([Xg.ravel(), Ug.ravel(), [T0]]), p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
            wv = np.array(r['x']).ravel(); T = wv[-1]; U = wv[6 * (N + 1):6 * (N + 1) + 3 * N].reshape(N, 3); xe, wmax, mc = simulate(x0, U, T)
            ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(wr(xe[:3] - C3)) < Rq + 1e-3) and np.all(np.abs(xe[3:]) < Rw + 1e-3) and wmax <= WM + 1e-2 and mc >= 0
            res.append((ok, T, br, s, st, time.time() - t0))
            if ok and (best is None or T < best[0]): best = (T, U, br)
    return best, res
