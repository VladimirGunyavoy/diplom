"""research-21: эталон минимального времени «статик кар» (SafeCarGoal journal-2026, `knowledge/research/static_car.md`).
(x, y, θ, v), ẋ = v cos θ, ẏ = v sin θ, θ̇ = ω, v̇ = a − .625 v; |a| ≤ 1.5, |ω| ≤ 1.8, v ∈ [−1, 2], |x|, |y| ≤ 5; 3 диска r .7; цель — круг (4, 3.6) r .45.
Multiple shooting CasADi/IPOPT, свободное T, u кусочно-пост. на N отрезках × M подшагов rk4; диски/стены — в КАЖДОМ подшаге; мультистарт — затравки
через промежуточную точку (прямо, обходы дисков с обеих сторон, случайные). Проверка — rk4 numpy dt .002 (касание диска / стены / цель).
Запуск (aida): PYTHONPATH=~/spore_v5/r5/pylib python3 ref_car.py N M i  → ref/car_q<i>.npy (T, U)"""
import os, sys, time, json
import numpy as np, casadi as ca
K, AM, WM, VL, VH, XL = .625, 1.5, 1.8, -1., 2., 5.
HZ = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); HR = .70; GOAL, GR = np.array([4., 3.6]), .45

def starts(n=20, seed=0):
    rng = np.random.default_rng(seed); S = []
    while len(S) < n:
        p = rng.uniform([-4.25, -4.25], [0., .3])
        if np.linalg.norm(p - GOAL) < GR + .8 or (np.linalg.norm(HZ - p, axis=1) - HR).min() <= .2: continue
        d = GOAL - p; S.append(np.array([p[0], p[1], np.arctan2(d[1], d[0]) + rng.normal(0, .25), 0.]))
    return S

def f_np(x, u): return np.array([x[3] * np.cos(x[2]), x[3] * np.sin(x[2]), u[1], u[0] - K * x[3]])
def f_ca(x, u): return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), u[1], u[0] - K * x[3])

def make_solver(N, M):
    X = ca.SX.sym('x', 4); U = ca.SX.sym('u', 2); h = ca.SX.sym('h'); xs = [X]; x = X
    for _ in range(M):
        k1 = f_ca(x, U); k2 = f_ca(x + h / 2 * k1, U); k3 = f_ca(x + h / 2 * k2, U); k4 = f_ca(x + h * k3, U); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); xs.append(x)
    F = ca.Function('F', [X, U, h], [x, ca.horzcat(*xs[1:])])
    opti = ca.Opti(); Xv = opti.variable(4, N + 1); Uv = opti.variable(2, N); T = opti.variable(); P = opti.parameter(4)
    opti.minimize(T); opti.subject_to(Xv[:, 0] == P); opti.subject_to(opti.bounded(.05, T, 30.))
    for i in range(N):
        xe, sub = F(Xv[:, i], Uv[:, i], T / N / M); opti.subject_to(Xv[:, i + 1] == xe)
        for j in range(M):
            for c in HZ: opti.subject_to((sub[0, j] - c[0]) ** 2 + (sub[1, j] - c[1]) ** 2 >= HR ** 2)
            opti.subject_to(opti.bounded(-XL, sub[0, j], XL)); opti.subject_to(opti.bounded(-XL, sub[1, j], XL)); opti.subject_to(opti.bounded(VL, sub[3, j], VH))
    opti.subject_to(opti.bounded(-AM, Uv[0, :], AM)); opti.subject_to(opti.bounded(-WM, Uv[1, :], WM))
    opti.subject_to((Xv[0, N] - GOAL[0]) ** 2 + (Xv[1, N] - GOAL[1]) ** 2 <= (GR - 1e-3) ** 2)
    opti.solver('ipopt', {'print_time': 0}, {'print_level': 0, 'max_iter': 3000, 'tol': 1e-8})
    return opti, Xv, Uv, T, P

def guess(x0, via, N):
    """затравка: ломаная старт → via → центр цели, скорость 1.5, курс по касательной"""
    pts = np.array([x0[:2], via, GOAL]); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = seg.sum(); s = np.linspace(0, L, N + 1)
    xy = np.c_[np.interp(s, np.r_[0, np.cumsum(seg)], pts[:, 0]), np.interp(s, np.r_[0, np.cumsum(seg)], pts[:, 1])]
    th = np.unwrap(np.r_[np.arctan2(*(np.diff(xy, axis=0)[:, ::-1].T)), np.arctan2(*(np.diff(xy, axis=0)[-1, ::-1]))]); th[0] = x0[2]
    return np.c_[xy, th, np.full(N + 1, 1.5)].T, L / 1.5

def simulate(x0, U, T, dt=.002):
    N = len(U); x = np.array(x0, float); n = max(1, int(np.ceil(T / N / dt))); h = T / N / n; dmin, wmin = np.inf, np.inf
    for u in U:
        for _ in range(n):
            k1 = f_np(x, u); k2 = f_np(x + h / 2 * k1, u); k3 = f_np(x + h / 2 * k2, u); k4 = f_np(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            dmin = min(dmin, (np.linalg.norm(HZ - x[:2], axis=1) - HR).min()); wmin = min(wmin, XL - np.abs(x[:2]).max())
    return x, dmin, wmin

def vias(x0, rng, nr=6):
    V = [(x0[:2] + GOAL) / 2]
    for c in HZ:
        d = GOAL - x0[:2]; nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        V += [c + nrm * (HR + .35), c - nrm * (HR + .35)]
    return V + [rng.uniform(-4.5, 4.5, 2) for _ in range(nr)]

if __name__ == '__main__':
    N, M = int(sys.argv[1]), int(sys.argv[2]); qi = [int(v) for v in sys.argv[3].split(',')]; Q = starts(); out = os.environ.get('OUT', 'ref'); os.makedirs(out, exist_ok=True)
    opti, Xv, Uv, Tv, P = make_solver(N, M)
    from tqdm import tqdm
    for i in tqdm(qi, desc='эталон OCP по стартам', mininterval=float(os.environ.get('TQDM_MI', 10))):
        t0 = time.time(); x0 = Q[i]; rng = np.random.default_rng(100 + i); res = []
        for via in vias(x0, rng):
            Xg, Tg = guess(x0, via, N); opti.set_value(P, x0); opti.set_initial(Xv, Xg); opti.set_initial(Tv, Tg); opti.set_initial(Uv, np.zeros((2, N)))
            try: sol = opti.solve(); T = float(sol.value(Tv)); U = np.array(sol.value(Uv)).T.reshape(N, 2)
            except Exception: continue
            xe, dmin, wmin = simulate(x0, U, T); ok = np.linalg.norm(xe[:2] - GOAL) <= GR + 1e-3 and dmin >= -1e-3 and wmin >= -1e-3
            res.append((ok, T, U, dmin))
        good = sorted([r for r in res if r[0]], key=lambda r: r[1])
        if good: np.save(os.path.join(out, 'car_q%d.npy' % i), np.r_[[good[0][1]], good[0][2].ravel()])
        print(json.dumps(dict(q=i, x0=np.round(x0, 3).tolist(), T_ref=round(good[0][1], 4) if good else None, ok=len(good), n=len(res),
                              best5=[round(r[1], 3) for r in good[:5]], dmin=round(good[0][3], 4) if good else None, sec=round(time.time() - t0, 1))), flush=True)
