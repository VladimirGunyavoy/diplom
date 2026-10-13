"""доводка эпизода до точной цели (решение research-27 по замечанию пользователя 2026-10-13 15:38): числа бенча не трогаем (T — вход в окно цели), но после последнего состояния
эпизода симуляция продолжается той же динамикой env.f с управлением «LQR по линеаризации в цели + компенсация равновесия, |τ| ≤ umax» (PD с компенсацией гравитации на dp1 не держит неустойчивую стойку при насыщении моментов — проверено); если LQR не довёл за 3 с — короткий OCP min T до |e|,|w| ≤ .008 (CasADi, aida). Остановка: |e| < TOL_Q и |w| < TOL_W (e — угловая ошибка, обёртка 2π) или TMAX с.
Хвост пишется в эпизод полем X_tail (кадры после последнего, тот же dt) и U_tail; T, X, U не меняются. Нужен bench/common.py (numpy), запускается на хабе; плеер (pygame+numpy+json) только читает.
python finish_tail.py dp1 [manip ...] — дописывает хвосты в viz/data/<система>.json (эпизоды с ok=true; прежние X_tail заменяются)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'bench'))
import numpy as np
KP, KD, TOL_Q, TOL_W, TMAX, H = 9., 6., .02, .02, 3., .01

def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def control(env, y, q_star, kp=KP, kd=KD):
    n2 = env.n // 2; y = np.asarray(y, float)[None]; z = np.zeros((1, env.m)); a0 = env.f(y, z)[0, n2:]
    B = np.stack([env.f(y, np.eye(env.m)[j][None])[0, n2:] - a0 for j in range(env.m)], 1); e = np.where(env.period[:n2] > 0, _wrap(y[0, :n2] - q_star), y[0, :n2] - q_star)
    u = np.linalg.solve(B, -kp * e - kd * y[0, n2:] - a0); return np.clip(u, env.umin, env.umax)
def lqr_gain(env, q_star):
    """LQR по линеаризации в цели (A, B численно; управление равновесия ug); scipy — только здесь"""
    import scipy.linalg as sl
    n, n2 = env.n, env.n // 2; y0 = np.zeros(n); y0[:n2] = q_star; f = lambda y, u: env.f(y[None], u[None])[0]; z = np.zeros(env.m); a0 = f(y0, z)[n2:]
    B0 = np.stack([f(y0, np.eye(env.m)[j])[n2:] - a0 for j in range(env.m)], 1); ug = np.linalg.solve(B0, -a0); eps = 1e-6
    A = np.stack([(f(y0 + eps * np.eye(n)[i], ug) - f(y0 - eps * np.eye(n)[i], ug)) / (2 * eps) for i in range(n)], 1)
    B = np.stack([(f(y0, ug + eps * np.eye(env.m)[j]) - f(y0, ug - eps * np.eye(env.m)[j])) / (2 * eps) for j in range(env.m)], 1)
    P = sl.solve_continuous_are(A, B, np.diag([100.] * n2 + [1.] * n2), np.eye(env.m)); return B.T @ P, ug
def _errs(env, y, q_star):
    n2 = env.n // 2; e = np.where(env.period[:n2] > 0, _wrap(y[:n2] - q_star), y[:n2] - q_star); return np.abs(e).max(), np.abs(y[n2:]).max()
def _play(env, y, us, dts, dt):
    """кусочно-постоянное (u, d) → кадры через dt (RK4 шагом ≤ .01)"""
    y = np.asarray(y, float)[None].copy(); X, U = [], []; t = 0.; nxt = dt
    for u, d in zip(us, dts):
        rem = float(d)
        while rem > 1e-12:
            st = min(rem, nxt - t); k = max(1, int(np.ceil(st / H - 1e-9)))
            for _ in range(k): y = env.step(y, np.asarray(u, float)[None], st / k)
            t += st; rem -= st
            if abs(t - nxt) < 1e-9: X.append(y[0].copy()); U.append(np.asarray(u, float)); nxt += dt
    return X, U, y[0]
def lqr_tail(env, y_last, q_star, dt):
    K, ug = lqr_gain(env, q_star); n2 = env.n // 2; y = np.asarray(y_last, float).copy(); X, U = [], []; t = 0.
    while t < TMAX - 1e-9:
        yy = y.copy(); yy[:n2] = np.where(env.period[:n2] > 0, _wrap(yy[:n2] - q_star), yy[:n2] - q_star) + q_star; u = np.clip(ug - K @ (yy - np.r_[q_star, np.zeros(n2)]), env.umin, env.umax)
        Xs, Us, y = _play(env, y, [u], [dt], dt); X += Xs; U += Us; t += dt
        if min(_errs(env, y, q_star)[0] / TOL_Q, 9) < 1 and _errs(env, y, q_star)[1] < TOL_W: break
    return np.array(X), np.array(U), t, y
def ocp_tail(env, y_last, q_star, dt, N=40):
    """минимум времени от y_last до |e|, |w| ≤ .01 (CasADi+IPOPT; динамика bench/ocp_casadi.dyn): когда LQR с |τ| ≤ umax не удерживает (неустойчивая стойка у края окна)"""
    import casadi as ca, ocp_casadi as OC
    n, n2 = env.n, env.n // 2; opti = ca.Opti(); T = opti.variable(); X = opti.variable(n, N + 1); U = opti.variable(env.m, N); h = T / N / 4
    def F(x, u):
        for _ in range(4): k1 = OC.dyn(env.name, x, u); k2 = OC.dyn(env.name, x + h / 2 * k1, u); k3 = OC.dyn(env.name, x + h / 2 * k2, u); k4 = OC.dyn(env.name, x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    y0 = np.asarray(y_last, float); y0[:n2] = np.where(env.period[:n2] > 0, _wrap(y0[:n2] - q_star), y0[:n2] - q_star) + q_star
    opti.subject_to(X[:, 0] == y0)
    for k in range(N): opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k]))
    for j in range(env.m): opti.subject_to(opti.bounded(float(env.umin[j]), U[j, :], float(env.umax[j])))
    for i in range(n2): opti.subject_to(opti.bounded(-.008, X[i, N] - q_star[i], .008)); opti.subject_to(opti.bounded(-.008, X[n2 + i, N], .008))
    opti.subject_to(opti.bounded(.1, T, 8.)); opti.minimize(T); opti.solver('ipopt', dict(print_time=False), dict(print_level=0, max_iter=400, tol=1e-7, sb='yes'))
    best = None
    for T0 in (1.5, 3., 5.):
        opti.set_initial(T, T0); a = np.linspace(0, 1, N + 1); opti.set_initial(X, y0[:, None] + (np.r_[q_star, np.zeros(n2)] - y0)[:, None] * a[None]); opti.set_initial(U, 0.)
        try: opti.solve()
        except RuntimeError: pass
        Tv = float(opti.debug.value(T)); Uv = np.asarray(opti.debug.value(U)).reshape(env.m, N).T
        if not (np.isfinite(Tv) and np.isfinite(Uv).all()): continue
        Xs, Us, yf = _play(env, y0, list(Uv), [Tv / N] * N, dt); eq, ew = _errs(env, yf, q_star)
        if eq < TOL_Q and ew < TOL_W and (best is None or Tv < best[0]): best = (Tv, Xs, Us, yf)
    return best
def tail(env, y_last, q_star=None, dt=.02):
    """→ X_tail (кадры через dt после y_last), U_tail, длительность, метод ('LQR'|'OCP'|None); None-метод — не довели (тогда хвост пуст)"""
    n2 = env.n // 2; q_star = np.zeros(n2) if q_star is None else np.asarray(q_star, float)
    X, U, t, y = lqr_tail(env, y_last, q_star, dt); eq, ew = _errs(env, y, q_star)
    if eq < TOL_Q and ew < TOL_W: return X, U, t, 'LQR'
    b = ocp_tail(env, y_last, q_star, dt)
    if b is not None: Tv, Xs, Us, yf = b; return np.array(Xs), np.array(Us), len(Xs) * dt, 'OCP'
    return np.zeros((0, env.n)), np.zeros((0, env.m)), 0., None
def add_tails(sysn, q_star=None):
    import common; env = common.make(sysn); fn = os.path.join(HERE, 'data', sysn + '.json'); d = json.load(open(fn)); rep = []
    for e in d['episodes']:
        e.pop('X_tail', None); e.pop('U_tail', None)
        if not e.get('ok', True): continue
        Xt, Ut, t, how = tail(env, e['X'][-1], q_star, e['dt'])
        if how is None: rep.append('%s s%s: НЕ ДОВЕЛИ' % (e['method'][:6], e['start'])); continue
        e['X_tail'] = np.round(Xt, 4).tolist(); e['U_tail'] = np.round(Ut, 3).tolist()
        n2 = env.n // 2; fin = Xt[-1]; eq = np.abs(_wrap(fin[:n2])).max(); rep.append('%s s%s: хвост %.2f с (%s), финал |e|=%.3f |w|=%.3f' % (e['method'][:6], e['start'], t, how, eq, np.abs(fin[n2:]).max()))
    json.dump(d, open(fn, 'w'), ensure_ascii=False, separators=(',', ':')); print('\n'.join(rep))
if __name__ == '__main__':
    for s in sys.argv[1:]: add_tails(s)
