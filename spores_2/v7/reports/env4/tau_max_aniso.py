"""ENV4 (B): τ_max локальной модели потока дрона (cell_model_size.py research hub-research-4, адаптация worker-34). 4 слоя-угла тяг, r = 0.3 / 0.6 / 0.9 (боковые направления в норм. координатах 1), 30 случайных спор,
tol 1e-3 / 1e-2, лин / квадр. Запуск: python3 tau_max_drone.py → печать JSON [p10, медиана, доля τ≥1]."""
import sys, json, numpy as np
SYS = 'drone'; rng = np.random.default_rng(0); TAUS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
def rk4(f, x, t, h=0.01):
    n = max(1, int(np.ceil(abs(t) / h))); hh = t / n
    for _ in range(n):
        k1 = f(x); k2 = f(x + hh / 2 * k1); k3 = f(x + hh / 2 * k2); k4 = f(x + hh * k3); x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
U = [(1., .5), (1., -.5), (-1., .5), (-1., -.5)]; L = 4
def F(y, k):
    u = U[k]; th, vx, vy, w = y[2], y[3], y[4], y[5]; c, s = np.cos(th), np.sin(th); sp = np.sqrt(vx ** 2 + vy ** 2 + 1e-12)
    return np.array([vx, vy, w, c * u[0] - s * u[1] - .05 * sp * vx, s * u[0] + c * u[1] - .05 * sp * vy - .5, .2 * u[1] / .1])
box = lambda: np.array([rng.uniform(-3, 3), rng.uniform(0, 4), rng.uniform(-np.pi, np.pi), *rng.uniform(-2, 2, 2), rng.uniform(-2, 2)])
def basis(v):
    d = len(v); Q, _ = np.linalg.qr(np.column_stack([v] + [np.eye(d)[i] for i in range(d)])); return Q[:, 1:d]
# анизотропный радиус: δ = R∘e в 6D (R = (r_xy, r_xy, r, r, r, r)), полный якобиан/гессиан по 6 координатам. Аргумент: r_xy [r_rest=.3]
rxy = float(sys.argv[1]); rr = float(sys.argv[2]) if len(sys.argv) > 2 else .3; R = np.array([rxy, rxy, rr, rr, rr, rr]); TAUS = np.round(np.arange(0.1, 1.0001, 0.1), 2); tol = 1e-2; N = 15
I6 = np.eye(6); dirs = [R * I6[i] * sg for i in range(6) for sg in (1, -1)] + [R * rng.uniform(-1, 1, 6) for _ in range(8)]
rows = []
for _ in range(N):
    c = box()
    for k in range(L):
        fk = lambda x: F(x, k); res = {}
        for tau in TAUS:
            P0 = rk4(fk, c, tau); h = 1e-4
            J = np.column_stack([(rk4(fk, c + h * I6[i], tau) - rk4(fk, c - h * I6[i], tau)) / (2 * h) for i in range(6)])
            hh = 2e-3; Hs = {}
            for i in range(6):
                for j in range(i, 6):
                    a, b = hh * I6[i], hh * I6[j]
                    Hs[i, j] = Hs[j, i] = (rk4(fk, c + a + b, tau) - rk4(fk, c + a - b, tau) - rk4(fk, c - a + b, tau) + rk4(fk, c - a - b, tau)) / (4 * hh * hh)
            eq = 0.0
            for d in dirs:
                tr = rk4(fk, c + d, tau); quad = P0 + J @ d + 0.5 * sum(Hs[i, j] * d[i] * d[j] for i in range(6) for j in range(6))
                eq = max(eq, np.linalg.norm(tr - quad))
            res[float(tau)] = eq
        ok = [t for t in TAUS if all(res[float(s)] <= tol for s in TAUS if s <= t)]; rows.append(max(ok) if ok else 0.0)
tm = np.array(rows); print('drone aniso r_xy=%.1f r_rest=%.2f tol=%g квадр: τ_max 10%%=%.2f медиана=%.2f доля τ=1: %.2f (N=%d)' % (rxy, rr, tol, np.percentile(tm, 10), np.median(tm), np.mean(tm >= 1.0), len(tm)), flush=True)
