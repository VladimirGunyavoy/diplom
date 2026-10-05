"""w18 (PLAN п.12, research-12): доводка коридора (u_k, h_k) пути атласа — multiple shooting, IPOPT (CasADi), g=2 висит→вверх.
Путь (Y старты дуг, U, H) из butterfly_dp.TRAJ (TRAJALL=1) режется на куски ≤ HMAX, дальше min ΣH при u∈[-UB,UB]², |ω|≤WM-.01 в каждом подшаге rk4,
окно цели с запасом .005. Допустимость — rk4 numpy dt .002. CLI: python3 -m src.cells7.refine path.npz  (Y,U,H) [out.npz]. aida: PYTHONPATH=~/spore_v5/r5/pylib."""
import os, sys, json, time
import numpy as np
import casadi as ca
G = float(os.environ.get('G', 2.)); UB = float(os.environ.get('UB', 1.)); WM = float(os.environ.get('WM', 3.)); HMIN = float(os.environ.get('HMIN', .02))
HMAX = float(os.environ.get('HMAX', .5)); M = int(os.environ.get('M', 40)); DTS = float(os.environ.get('DTS', .0125)); MS = [2.5, 1.0]; S11, S12, S22 = 2.5, 1., 1.
C3 = np.array([np.pi / 2, 0.]); RQ, RW = .3, .5; X0 = np.array([-np.pi / 2, 0, 0, 0]); OCP = 7.636
def acc(q1, q2, w1, w2, u1, u2, cos=np.cos, sin=np.sin):
    th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = cos(th1 - th2); s = sin(th1 - th2)
    Q1 = u1 - u2 - G * MS[0] * cos(th1) - S12 * s * d2 ** 2; Q2 = u2 - G * MS[1] * cos(th2) + S12 * s * d1 ** 2
    det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
    return a1, a2 - a1
def fnp(z, u): a1, a2 = acc(*z, *u); return np.array([z[2], z[3], a1, a2])
def rk4(fx, z, u, h):
    k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def simulate(U, H, dt=.002):
    z = X0.copy(); wm = 0.
    for u, h in zip(U, H):
        n = max(1, int(np.ceil(h / dt)))
        for _ in range(n): z = rk4(fnp, z, u, h / n); wm = max(wm, np.abs(z[2:]).max())
    return z, wm
def make(K):
    x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 2); h = ca.SX.sym('h')
    fx = lambda z, v: ca.vertcat(z[2], z[3], *acc(z[0], z[1], z[2], z[3], v[0], v[1], ca.cos, ca.sin)); F = ca.Function('F', [x, u, h], [rk4(fx, x, u, h)])
    X = ca.SX.sym('X', 4, K * M + 1); U = ca.SX.sym('U', 2, K); H = ca.SX.sym('H', K); P = ca.SX.sym('P', 2); g = [X[:, 0] - X0]
    for k in range(K):
        for j in range(M): i = k * M + j; g.append(F(X[:, i], U[:, k], H[k] / M) - X[:, i + 1])
    g.append(X[:2, -1] - P); w = ca.vertcat(ca.vec(X), ca.vec(U), H)
    S = ca.nlpsol('S', 'ipopt', {'x': w, 'f': ca.sum1(H), 'g': ca.vertcat(*g), 'p': P}, {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 3000, 'ipopt.tol': 1e-8})
    nx = 4 * (K * M + 1); lbx = np.full(nx + 3 * K, -np.inf); ubx = -lbx.copy(); Xi = np.arange(nx).reshape(-1, 4)
    lbx[Xi[:, 2:].ravel()] = -(WM - .01); ubx[Xi[:, 2:].ravel()] = WM - .01; lbx[Xi[-1, 2:]] = -(RW - .005); ubx[Xi[-1, 2:]] = RW - .005
    lbx[nx:nx + 2 * K] = -UB; ubx[nx:nx + 2 * K] = UB; lbx[nx + 2 * K:] = HMIN; ubx[nx + 2 * K:] = HMAX
    ng = 4 * (K * M + 1) + 2; lbg = np.zeros(ng); ubg = np.zeros(ng); lbg[-2:] = -(RQ - .005); ubg[-2:] = RQ - .005
    return S, lbx, ubx, lbg, ubg, nx
def cut(Y, U, H):
    """дуги ≤ HMAX (длинные — на равные куски того же u); M подшагов ≤ DTS."""
    Yo, Uo, Ho = [], [], []
    for y, u, h in zip(Y, U, H):
        n = int(np.ceil(h / HMAX - 1e-9)); z = np.array(y, float)
        for _ in range(n):
            Yo.append(z.copy()); Uo.append(u); Ho.append(h / n)
            for _ in range(20): z = rk4(fnp, z, u, h / n / 20)
    return np.array(Yo), np.clip(np.array(Uo), -UB, UB), np.maximum(np.array(Ho), HMIN)
def refine(Y, U, H):
    """→ (T, U, H, wmax, ok, статус, сек). T=inf, если не сошлось/недопустимо."""
    t0 = time.time(); Yw, Uw, Hw = cut(Y, U, H); K = len(Hw); S, lbx, ubx, lbg, ubg, nx = make(K)
    Xg = [X0.copy()]
    for i, (u, h) in enumerate(zip(Uw, Hw)):
        z = Yw[i].copy(); z[:2] = Xg[-1][:2] + ((z[:2] - Xg[-1][:2] + np.pi) % (2 * np.pi) - np.pi)
        for _ in range(M): z = rk4(fnp, z, u, h / M); Xg.append(z)
    Xg = np.array(Xg); end = Xg[-1, :2]; tgt = end + ((C3 - end + np.pi) % (2 * np.pi) - np.pi); Xg[:, 2:] = np.clip(Xg[:, 2:], -WM + .02, WM - .02)
    r = S(x0=np.concatenate([Xg.ravel(), Uw.ravel(), Hw]), p=tgt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
    w = np.array(r['x']).ravel(); Uo = w[nx:nx + 2 * K].reshape(K, 2); Ho = w[nx + 2 * K:]; ze, wm = simulate(Uo, Ho)
    dq = (ze[:2] - C3 + np.pi) % (2 * np.pi) - np.pi
    ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(dq) <= RQ + 1e-3) and np.all(np.abs(ze[2:]) <= RW + 1e-3) and wm <= WM + 1e-3
    return (float(Ho.sum()) if ok else np.inf), Uo, Ho, float(wm), bool(ok), st, time.time() - t0
if __name__ == '__main__':
    d = np.load(sys.argv[1]); T0 = float(d['H'].sum()); T, U, H, wm, ok, st, sec = refine(d['Y'], d['U'], d['H'])
    print(json.dumps(dict(path=sys.argv[1].split('/')[-1], arcs=len(d['H']), T_before=round(T0, 3), T=round(T, 4), T_over_OCP=round(T / OCP, 4), ok=ok, wmax=round(wm, 3), st=st[:14], sec=round(sec, 1))), flush=True)
    if len(sys.argv) > 2 and ok: np.savez(sys.argv[2], U=U, H=H)
