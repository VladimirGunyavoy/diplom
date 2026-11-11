"""ENV2 (b14, 2026-11-11): динамик кар journal-2026 (safe_car_goal_dynamic.py). Та же машинка (x, y, θ, v), (a, ω); три диска r .70 по их ЗАКОНУ (детерминированный):
t = 2π·step/180 (step = время/.08), offset_x = A_x sin(t+φ), offset_y = A_y cos(.73 t + 1.7 φ), A = .42·(±1)·(1, .65) (знак чередуется), φ_i = .35 + 2πi/3; центры клипаются внутрь арены ±5.
Режим random: φ случайна, A × U(.65, 1.2). Касание проверяется НЕПРЕРЫВНО (мелкий шаг по времени). Эталон-ясновидец: OCP CasADi со знанием траекторий дисков — нижняя граница T.
Базовая строка спор: атлас carfree (V₀ без дисков) + агент открытым планом — доля эпизодов с касанием (точка отсчёта, ожидаемо плохо)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
K, AM, WM, VL, VH, XL = .625, 1.5, 1.8, -1., 2., 5.
C0 = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); HR = .70; GOAL = np.array([4., 3.6]); GR = .45; DT = .08; PERIOD_S = 180 * DT   # 14.4 с

class Hazards:
    def __init__(s, mode='fixed', seed=0):
        rng = np.random.default_rng(seed); s.mode = mode
        s.sgn = np.array([1., -1., 1.]); s.A = float(os.environ.get('AMP', 1.)) * .42 * s.sgn[:, None] * np.array([1., .65])[None]
        s.phi = .35 + 2 * np.pi * np.arange(3) / 3
        if mode == 'random': s.phi = rng.uniform(0, 2 * np.pi, 3); s.A = s.A * rng.uniform(.65, 1.2, (3, 1))
    def pos(s, t):
        """центры дисков в момент t (с) — (3, 2); t может быть массивом → (len, 3, 2)"""
        t = np.asarray(t, float); ph = 2 * np.pi * (t / DT) / 180 * float(os.environ.get('FAST', 1.))
        ox = s.A[:, 0] * np.sin(ph[..., None] + s.phi); oy = s.A[:, 1] * np.cos(.73 * ph[..., None] + 1.7 * s.phi)
        c = np.stack([C0[:, 0] + ox, C0[:, 1] + oy], -1); return np.clip(c, -(XL - HR), XL - HR)
    def vel(s, t, h=1e-3): return (s.pos(t + h) - s.pos(t - h)) / (2 * h)

def starts(n=20, seed=0, H=None):
    """те же 20 стартов, что у статика (≥ .2 от СТАТИЧНЫХ позиций дисков; v = 0, θ на цель + N(0, .25))"""
    rng = np.random.default_rng(seed); S = []
    while len(S) < n:
        p = rng.uniform([-4.25, -4.25], [0., .3])
        if np.linalg.norm(p - GOAL) < GR + .8 or (np.linalg.norm(C0 - p, axis=1) - HR).min() <= .2: continue
        d = GOAL - p; S.append(np.array([p[0], p[1], np.arctan2(d[1], d[0]) + rng.normal(0, .25), 0.]))
    return np.array(S)

def replay(H, x0, us, dts, h=.01, sat=True):
    """непрерывный проигрыш плана (u_k, dt_k) RK4 шаг h, v насыщается в [−1, 2] (их среда). Возврат dict(ok, T, touch_frac_t (доля времени внутри диска), n_touch_ep (1 если было касание), pen)"""
    def fl(y, a, w):
        dv = a - K * y[3]
        if sat and ((y[3] >= VH and dv > 0) or (y[3] <= VL and dv < 0)): dv = 0.
        return np.array([y[3] * np.cos(y[2]), y[3] * np.sin(y[2]), w, dv])
    y = np.array(x0, float); t = 0.; pen = 0.; tin = 0.; nst = 0
    for (a, w), dt in zip(us, dts):
        k = max(1, int(np.ceil(dt / h - 1e-9))); hh = dt / k
        for _ in range(k):
            k1 = fl(y, a, w); k2 = fl(y + hh / 2 * k1, a, w); k3 = fl(y + hh / 2 * k2, a, w); k4 = fl(y + hh * k3, a, w); y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += hh; nst += 1
            d = (np.linalg.norm(H.pos(t) - y[:2], axis=1) - HR).min(); pen = max(pen, -d, np.abs(y[:2]).max() - XL); tin += hh * (d < 0)
            if np.linalg.norm(y[:2] - GOAL) <= GR: return dict(ok=True, T=t, touch=bool(pen > 1e-3), pen=float(max(pen, 0.)), t_inside=tin)
    return dict(ok=False, T=None, touch=bool(pen > 1e-3), pen=float(max(pen, 0.)), t_inside=tin)

def make_oracle(N=60, M=3):
    """OCP CasADi: диски — функции времени (t_k = T·(k)/(N·M)), мультистарт по затравкам. Возврат solve(x0, H, guesses) → (T, U) или None"""
    import casadi as ca
    def f_ca(x, u): return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), u[1], u[0] - K * x[3])
    X = ca.SX.sym('x', 4); U = ca.SX.sym('u', 2); h = ca.SX.sym('h'); x = X; xs = []
    for _ in range(M):
        k1 = f_ca(x, U); k2 = f_ca(x + h / 2 * k1, U); k3 = f_ca(x + h / 2 * k2, U); k4 = f_ca(x + h * k3, U); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); xs.append(x)
    F = ca.Function('F', [X, U, h], [x, ca.horzcat(*xs)])
    def solve(x0, H, via_list, tmax=30.):
        best = None
        for via in via_list:
            opti = ca.Opti(); Xv = opti.variable(4, N + 1); Uv = opti.variable(2, N); T = opti.variable(); opti.minimize(T); opti.subject_to(Xv[:, 0] == np.asarray(x0)); opti.subject_to(opti.bounded(.05, T, tmax))
            A = H.A; ph = H.phi
            for i in range(N):
                xe, sub = F(Xv[:, i], Uv[:, i], T / N / M); opti.subject_to(Xv[:, i + 1] == xe)
                for j in range(M):
                    tt = T * (i * M + j + 1) / (N * M); pp = 2 * np.pi * (tt / DT) / 180
                    for q in range(3):
                        cx = C0[q, 0] + A[q, 0] * ca.sin(pp + ph[q]); cy = C0[q, 1] + A[q, 1] * ca.cos(.73 * pp + 1.7 * ph[q])
                        cx = ca.fmin(ca.fmax(cx, -(XL - HR)), XL - HR); cy = ca.fmin(ca.fmax(cy, -(XL - HR)), XL - HR)
                        opti.subject_to((sub[0, j] - cx) ** 2 + (sub[1, j] - cy) ** 2 >= HR ** 2)
                    opti.subject_to(opti.bounded(-XL, sub[0, j], XL)); opti.subject_to(opti.bounded(-XL, sub[1, j], XL)); opti.subject_to(opti.bounded(VL, sub[3, j], VH))
            opti.subject_to(opti.bounded(-AM, Uv[0, :], AM)); opti.subject_to(opti.bounded(-WM, Uv[1, :], WM)); opti.subject_to((Xv[0, N] - GOAL[0]) ** 2 + (Xv[1, N] - GOAL[1]) ** 2 <= (GR - 1e-3) ** 2)
            pts = np.array([np.asarray(x0)[:2], via, GOAL]); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = seg.sum(); s_ = np.linspace(0, L, N + 1); cs = np.r_[0, np.cumsum(seg)]
            xy = np.c_[np.interp(s_, cs, pts[:, 0]), np.interp(s_, cs, pts[:, 1])]; th = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])); th = np.unwrap(np.r_[th, th[-1]]); th[0] = x0[2]
            opti.set_initial(Xv, np.c_[xy, th, np.full(N + 1, 1.5)].T); opti.set_initial(T, L / 1.5); opti.set_initial(Uv, np.zeros((2, N)))
            opti.solver('ipopt', {'print_time': 0}, {'print_level': 0, 'max_iter': 3000, 'tol': 1e-8})
            try: sol = opti.solve(); Tv = float(sol.value(T)); Us = np.array(sol.value(Uv)).T.reshape(N, 2)
            except Exception: continue
            if best is None or Tv < best[0]: best = (Tv, Us)
        return best
    return solve

def vias(x0, rng, nr=6):
    V = [(np.asarray(x0)[:2] + GOAL) / 2]
    for c in C0:
        d = GOAL - np.asarray(x0)[:2]; nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d); V += [c + nrm * (HR + .35), c - nrm * (HR + .35)]
    return V + [rng.uniform(-4.5, 4.5, 2) for _ in range(nr)]
