"""research hub-v5chain-research-8: перепроверка шага 7 («спектр+T у финиша») БЕЗ читерства: время до цели — V* клеток v7 (Cover), а не формула T*.
T* — только для оценки результата. DI, dt .06, 60 стартов, после входа в цель 3 с удержания."""
import numpy as np, json, sys, time
sys.path.insert(0, '.'); from v7_spore_di import Cover; from v7_faces_di import tstar_box
dt = .06; rng = np.random.default_rng(2); Q0 = rng.uniform(-1.5, 1.5, (60, 2))
def phi(x, v, u, h): return x + v * h + u * h * h / 2, v + u * h
for rho in (.1, .02):
    t0 = time.time(); Cv = Cover(tau=.4, r=.1, seeds=6000, rho=rho); Cv.solve(); print('cover', rho, Cv.K, round(time.time() - t0), flush=True)
    Vf = lambda X, Vv: np.nan_to_num(Cv.Vstar(np.stack([X, Vv], 1)), posinf=1e3)
    for nm, U in (('вершины', np.array([-1., 1.])), ('спектр+V', np.linspace(-1, 1, 41))):
        T_arr, hold, sws = [], [], []
        for x0, v0 in Q0:
            x, v, t, arrived, ta, mx, sw, pu = x0, v0, 0., False, np.nan, 0., 0, None
            while t < 15 + (3 if arrived else 0):
                X1, V1 = phi(x, v, U, dt); u = U[np.argmin(dt + Vf(X1, V1))]
                if pu is not None and abs(u - pu) > .5: sw += 1
                pu = u
                for _ in range(4): x, v = phi(x, v, u, dt / 4); t += dt / 4
                inside = abs(x) <= rho and abs(v) <= rho
                if inside and not arrived: arrived, ta, t_end = True, t, t + 3
                if arrived: mx = max(mx, np.hypot(x, v))
                if arrived and t >= t_end: break
            Ts = float(tstar_box(np.array([x0]), np.array([v0]), rho)[0])
            T_arr.append(ta / Ts if arrived and Ts > .05 else np.nan); hold.append(mx if arrived else np.nan); sws.append(sw)
        T_arr, hold = np.array(T_arr), np.array(hold)
        print(json.dumps(dict(rho=rho, agent=nm, V='клетки', arrived=round(float(np.isfinite(hold).mean()), 3), T_over_Tstar=round(float(np.nanmean(T_arr)), 4),
                              hold_max_med=float('%.2g' % np.nanmedian(hold)) if np.isfinite(hold).any() else None, sw_med=float(np.median(sws))), ensure_ascii=False), flush=True)
