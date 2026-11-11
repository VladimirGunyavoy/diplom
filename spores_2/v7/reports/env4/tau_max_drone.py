"""ENV4 (B): τ_max локальной модели потока дрона (cell_model_size.py research hub-research-4, адаптация worker-34). 4 слоя-угла тяг, r = 0.3 / 0.6 / 0.9 (боковые направления в норм. координатах 1), 30 случайных спор,
tol 1e-3 / 1e-2, лин / квадр. Запуск: python3 tau_max_drone.py → печать JSON [p10, медиана, доля τ≥1]."""
import sys, json, numpy as np
rng = np.random.default_rng(0); TAUS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
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
out = {}
for r in (0.3, 0.6, 0.9):
    rows = []
    for _ in range(30):
        c = box()
        for k in range(L):
            fk = lambda x: F(x, k); v = fk(c)
            if np.linalg.norm(v) < 1e-6: continue
            E = basis(v / np.linalg.norm(v)); m = E.shape[1]; dirs = [E[:, i] * sg for i in range(m) for sg in (1, -1)]
            for _ in range(8): w = E @ rng.normal(size=m); dirs.append(w / np.linalg.norm(w))
            res = {}
            for tau in TAUS:
                P0 = rk4(fk, c, tau); h = 1e-4
                J = np.column_stack([(rk4(fk, c + h * E[:, i], tau) - rk4(fk, c - h * E[:, i], tau)) / (2 * h) for i in range(m)])
                hh = 2e-3; Hs = {}
                for i in range(m):
                    for j in range(i, m):
                        a, b = hh * E[:, i], hh * E[:, j]
                        Hs[i, j] = Hs[j, i] = (rk4(fk, c + a + b, tau) - rk4(fk, c + a - b, tau) - rk4(fk, c - a + b, tau) + rk4(fk, c - a - b, tau)) / (4 * hh * hh)
                el = eq = 0.0
                for u in dirs:
                    z = r * (E.T @ u); tr = rk4(fk, c + r * u, tau); lin = P0 + J @ z
                    quad = lin + 0.5 * sum(Hs[i, j] * z[i] * z[j] for i in range(m) for j in range(m))
                    el = max(el, np.linalg.norm(tr - lin)); eq = max(eq, np.linalg.norm(tr - quad))
                res[float(tau)] = (el, eq)
            rows.append(res)
    for tol in (1e-3, 1e-2):
        for mi, nm in ((0, 'лин'), (1, 'квадр')):
            tm = []
            for res in rows:
                ok = [t for t in TAUS if all(res[float(s)][mi] <= tol for s in TAUS if s <= t)]; tm.append(max(ok) if ok else 0.0)
            tm = np.array(tm); out['r%.1f_tol%g_%s' % (r, tol, nm)] = [float(np.percentile(tm, 10)), float(np.median(tm)), float(np.mean(tm >= 1.0))]
            print('%s r=%.1f tol=%g %s: τ_max 10%%=%.2f медиана=%.2f доля τ=1: %.2f' % (SYS, r, tol, nm, *out['r%.1f_tol%g_%s' % (r, tol, nm)]), flush=True)
json.dump(out, open('cell_model_size_%s.json' % SYS, 'w'), indent=1)
