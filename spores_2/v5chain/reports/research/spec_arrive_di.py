"""research hub-research-7: прибытие и удержание без LQR (вариант пользователя 2026-10-02): агент argmin_u [dt + T*(φ_u(x,dt))], u из спектра,
против спектр+LQR (P из DARE, Q по Брайсону = diag(1/ρ²), R = 1) и вершин ±1. DI, T* точное (до квадрата ρ), dt .06; после входа в цель — 3 с удержания."""
import numpy as np, json, sys
from scipy.linalg import solve_discrete_are
sys.path.insert(0, '.'); from v7_faces_di import tstar_box, tstar_pt
dt = .06; rng = np.random.default_rng(2); Q0 = rng.uniform(-1.5, 1.5, (60, 2))
def phi(x, v, u, h): return x + v * h + u * h * h / 2, v + u * h
def Tgo(x, v, rho): return tstar_pt(x, v, 0., 0.) if rho == 0 else tstar_box(x, v, rho, n=401)
res = []
for rho in (.1, .02, .005, 0.):
    A = np.array([[1, dt], [0, 1]]); B = np.array([[dt * dt / 2], [dt]]); r_ = max(rho, .001); P = solve_discrete_are(A, B, np.diag([1 / r_**2] * 2), np.eye(1))
    for nm, U in (('вершины', np.array([-1., 1.])), ('спектр+T', np.linspace(-1, 1, 41)), ('спектр+LQR', np.linspace(-1, 1, 41))):
        T_arr, hold, sws = [], [], []
        for x0, v0 in Q0:
            x, v, t, arrived, ta, mx, sw, pu = x0, v0, 0., False, np.nan, 0., 0, None
            while t < 15 + (3 if arrived else 0):
                X1, V1 = phi(x, v, U, dt)
                near = abs(x) < 10 * r_ and abs(v) < 10 * r_
                c = (dt + Tgo(X1, V1, rho)) if (nm != 'спектр+LQR' or not near) else np.einsum('ni,ij,nj->n', np.c_[X1, V1], P, np.c_[X1, V1])
                u = U[np.argmin(c)]
                if pu is not None and abs(u - pu) > .5: sw += 1
                pu = u
                for _ in range(4): x, v = phi(x, v, u, dt / 4); t += dt / 4
                inside = abs(x) <= max(rho, 1e-3) and abs(v) <= max(rho, 1e-3)
                if inside and not arrived: arrived, ta, t_end = True, t, t + 3
                if arrived: mx = max(mx, np.hypot(x, v))
                if arrived and t >= t_end: break
            Ts = float(Tgo(np.array([x0]), np.array([v0]), rho)[0]) if rho else float(tstar_pt(x0, v0, 0., 0.))
            T_arr.append(ta / Ts if arrived and Ts > .05 else np.nan); hold.append(mx if arrived else np.nan); sws.append(sw)
        T_arr, hold = np.array(T_arr), np.array(hold)
        d = dict(rho=rho, agent=nm, arrived=round(float(np.isfinite(hold).mean()), 3), T_over_Tstar=round(float(np.nanmean(T_arr)), 4),
                 hold_max_med=float('%.2g' % np.nanmedian(hold)) if np.isfinite(hold).any() else None, sw_med=float(np.median(sws)))
        print(json.dumps(d, ensure_ascii=False), flush=True); res.append(d)
json.dump(res, open('spec_arrive_di.json', 'w'), ensure_ascii=False, indent=1)
