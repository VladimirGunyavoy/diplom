"""ENV4 подводный дрон (journal-2026 publish/env/underwaterdrone.py; код только читали): эталон минимального времени OCP.
Состояние (x, y, θ, vx, vy, ω); u = (F_long ±1, F_lat ±.5); m=1, I=.1, Cd=.05 (F=Cd|v|v), g=.5, плечо .2 (τ = .2·F_lat), |v|,|ω| ≤ 3; поле x∈[−3,3], y∈[0,4];
цель: y ≥ 4 и |x| ≤ .4; запрет: эллипс (−3,2) a4 b1 (центр дрона в нём) + круг (1,2) r.75. Непрерывная модель (их шаг Эйлера .02 — отдельная проверка).
Старты: seed 0, x∈[−2.75,2.75], y∈[.2,3.6], вне запретных зон с зазором .1, θ∈U(0,2π), v,ω ~ N(0,.2) (штатный random_start).
Multiple shooting CasADi/IPOPT, свободное T, u кусочно-постоянно на N отрезках × M подшагов rk4, препятствия в каждом подшаге; несколько затравок (обход справа/слева/прямо/случайные), берём лучшее.
Запуск (aida, из bench/): PYTHONPATH=~/spore_v5/r5/pylib python3 env4_ref.py N M i,j,..|all → OUT/env4_q<i>.npz (T, U, X); OUT/env4_ref.json"""
import os, sys, time, json
import numpy as np, casadi as ca
M_, I_, CD, G_, LAT = 1., .1, .05, .5, .2; FL, FT = 1., .5; VMAX, WMAX = 3., 3.; XM, YT, HW = 3., 4., .4
EC, EA, EB = np.array([-3., 2.]), 4., 1.; CC, CR = np.array([1., 2.]), .75

def clr(x, y):   # зазор среды (эллипс по их формуле, круг — расстояние − r)
    dx, dy = x - EC[0], y - EC[1]; rho = np.hypot(dx / EA, dy / EB)
    ce = -min(EA, EB) if rho < 1e-9 else (rho - 1) * np.hypot(EA * dx / (EA * rho), EB * dy / (EB * rho))
    return min(ce, np.hypot(x - CC[0], y - CC[1]) - CR)

def starts(n=int(os.environ.get('NSTARTS', 20)), seed=int(os.environ.get('STARTSEED', 0))):
    rng = np.random.default_rng(seed); S = []
    while len(S) < n:
        x, y = rng.uniform(-2.75, 2.75), rng.uniform(.2, 3.6)
        if clr(x, y) < .1: continue
        S.append(np.array([x, y, rng.uniform(0, 2 * np.pi), *rng.normal(0, .2, 3)]))
    return S

def f_ca(s, u):
    th, vx, vy, w = s[2], s[3], s[4], s[5]; c, sn = ca.cos(th), ca.sin(th); sp = ca.sqrt(vx ** 2 + vy ** 2 + 1e-8)
    ax = (c * u[0] - sn * u[1] - CD * sp * vx) / M_; ay = (sn * u[0] + c * u[1] - CD * sp * vy - G_) / M_
    return ca.vertcat(vx, vy, w, ax, ay, LAT * u[1] / I_)

def f_np(s, u):
    th, vx, vy, w = s[2], s[3], s[4], s[5]; c, sn = np.cos(th), np.sin(th); sp = np.hypot(vx, vy)
    return np.array([vx, vy, w, (c * u[0] - sn * u[1] - CD * sp * vx) / M_, (sn * u[0] + c * u[1] - CD * sp * vy - G_) / M_, LAT * u[1] / I_])

def make_solver(N, M):
    X = ca.SX.sym('x', 6); U = ca.SX.sym('u', 2); h = ca.SX.sym('h'); x = X; xs = []
    for _ in range(M):
        k1 = f_ca(x, U); k2 = f_ca(x + h / 2 * k1, U); k3 = f_ca(x + h / 2 * k2, U); k4 = f_ca(x + h * k3, U); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); xs.append(x)
    F = ca.Function('F', [X, U, h], [x, ca.horzcat(*xs)])
    opti = ca.Opti(); Xv = opti.variable(6, N + 1); Uv = opti.variable(2, N); T = opti.variable(); P = opti.parameter(6)
    opti.minimize(T); opti.subject_to(Xv[:, 0] == P); opti.subject_to(opti.bounded(.1, T, 40.))
    for i in range(N):
        xe, sub = F(Xv[:, i], Uv[:, i], T / N / M); opti.subject_to(Xv[:, i + 1] == xe)
        for j in range(M):
            px, py = sub[0, j], sub[1, j]
            opti.subject_to(((px - EC[0]) / EA) ** 2 + ((py - EC[1]) / EB) ** 2 >= 1.); opti.subject_to((px - CC[0]) ** 2 + (py - CC[1]) ** 2 >= CR ** 2)
            opti.subject_to(opti.bounded(-XM, px, XM)); opti.subject_to(opti.bounded(0., py, YT)); opti.subject_to(opti.bounded(-VMAX, sub[3, j], VMAX)); opti.subject_to(opti.bounded(-VMAX, sub[4, j], VMAX)); opti.subject_to(opti.bounded(-WMAX, sub[5, j], WMAX))
    opti.subject_to(opti.bounded(-FL, Uv[0, :], FL)); opti.subject_to(opti.bounded(-FT, Uv[1, :], FT))
    opti.subject_to(Xv[1, N] >= YT - 1e-6); opti.subject_to(opti.bounded(-HW + 1e-3, Xv[0, N], HW - 1e-3))
    opti.solver('ipopt', {'print_time': 0}, {'print_level': 0, 'max_iter': 3000, 'tol': 1e-7})
    return opti, Xv, Uv, T, P

def guess(x0, via, N):
    pts = np.array([x0[:2], *via, [0., YT]]); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = seg.sum(); s = np.linspace(0, L, N + 1); cs = np.r_[0, np.cumsum(seg)]
    xy = np.c_[np.interp(s, cs, pts[:, 0]), np.interp(s, cs, pts[:, 1])]; Tg = max(L / 1.2, 1.)
    X = np.zeros((6, N + 1)); X[:2] = xy.T; X[2] = x0[2]; X[3:] = x0[3:, None]; return X, Tg

def vias(x0, rng, nr=5):
    V = [[(2.4, 2.0)], [(2.4, .6), (2.4, 3.2)], [(0., .5), (2.4, 2.)], [(0., 3.5)]]
    return V + [[rng.uniform([-2.8, .2], [2.8, 3.8])] for _ in range(nr)]

def simulate(x0, U, T, dt=.001):   # проверка: rk4 numpy, касание = clr < 0; конец — первое y ≥ 4 внутри окна
    N = len(U); x = np.array(x0, float); n = max(1, int(np.ceil(T / N / dt))); h = T / N / n; cmin = np.inf; tt = 0.
    for u in U:
        for _ in range(n):
            k1 = f_np(x, u); k2 = f_np(x + h / 2 * k1, u); k3 = f_np(x + h / 2 * k2, u); k4 = f_np(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); cmin = min(cmin, clr(x[0], x[1]))
    return x, cmin

if __name__ == '__main__':
    N, M = int(sys.argv[1]), int(sys.argv[2]); Q = starts(); qi = list(range(len(Q))) if sys.argv[3] == 'all' else [int(v) for v in sys.argv[3].split(',')]; out = os.environ.get('OUT', 'ref'); os.makedirs(out, exist_ok=True)
    opti, Xv, Uv, Tv, P = make_solver(N, M); from tqdm import tqdm; summary = {}
    for i in tqdm(qi, desc='эталон OCP дрона', mininterval=float(os.environ.get('TQDM_MI', 10))):
        t0 = time.time(); x0 = Q[i]; rng = np.random.default_rng(100 + i); best = None
        for via in vias(x0, rng):
            Xg, Tg = guess(x0, via, N); opti.set_value(P, x0); opti.set_initial(Xv, Xg); opti.set_initial(Tv, Tg); opti.set_initial(Uv, np.zeros((2, N)))
            try: sol = opti.solve(); T = float(sol.value(Tv)); U = np.array(sol.value(Uv)).T.reshape(N, 2); X = np.array(sol.value(Xv))
            except Exception: continue
            xf, cm = simulate(x0, U, T)
            if cm < -1e-3 or xf[1] < YT - .02 or abs(xf[0]) > HW + .02: continue
            if best is None or T < best[0]: best = (T, U, X, cm, xf)
        if best is None: summary[i] = None; print('старт', i, 'эталон не найден', flush=True); continue
        np.savez(os.path.join(out, 'env4_q%d.npz' % i), T=best[0], U=best[1], X=best[2], x0=x0); summary[i] = dict(T=best[0], clr_min=best[3], xf=best[4].tolist(), sec=time.time() - t0, x0=x0.tolist())
        print('старт', i, 'T %.3f clr_min %.3f %.0f с' % (best[0], best[3], time.time() - t0), flush=True)
    json.dump(summary, open(os.path.join(out, 'env4_ref_%s.json' % sys.argv[3].replace(',', '_')), 'w'))
