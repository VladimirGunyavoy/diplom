"""ENV4 (B): τ_max локальной модели потока дрона (cell_model_size.py research hub-research-4, адаптация worker-34). 4 слоя-угла тяг, r = 0.3 / 0.6 / 0.9 (боковые направления в норм. координатах 1), 30 случайных спор,
tol 1e-3 / 1e-2, лин / квадр. Запуск: python3 tau_max_drone.py → печать JSON [p10, медиана, доля τ≥1]."""
import sys, json, numpy as np
SYS = 'car'; rng = np.random.default_rng(0); TAUS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
def rk4(f, x, t, h=0.01):
    n = max(1, int(np.ceil(abs(t) / h))); hh = t / n
    for _ in range(n):
        k1 = f(x); k2 = f(x + hh / 2 * k1); k3 = f(x + hh / 2 * k2); k4 = f(x + hh * k3); x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
KV = .625; UA, UW = 1.5, 1.8; U = [(a, w) for a in (UA, -UA) for w in (UW, -UW)] + [(UA, 0.), (-UA, 0.)]; L = 6; D = 4
def F(y, k):
    u = U[k]; return np.array([y[3] * np.cos(y[2]), y[3] * np.sin(y[2]), u[1], u[0] - KV * y[3]])
box = lambda: np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi), rng.uniform(-1, 2)])
out = {}
# анизотропный радиус: δ = R∘e в 6D (R = (r_xy, r_xy, r, r, r, r)), полный якобиан/гессиан по 6 координатам. Аргумент: r_xy [r_rest=.3]
rxy = float(sys.argv[1]); rr = float(sys.argv[2]) if len(sys.argv) > 2 else .3; R = np.array([rxy, rxy, rr, rr]); TAUS = np.round(np.arange(0.1, 1.0001, 0.1), 2); tol = 1e-2; N = 15
I6 = np.eye(D); dirs = [R * I6[i] * sg for i in range(D) for sg in (1, -1)] + [R * rng.uniform(-1, 1, D) for _ in range(8)]
rows = []
for _ in range(N):
    c = box()
    for k in range(L):
        fk = lambda x: F(x, k); res = {}
        for tau in TAUS:
            P0 = rk4(fk, c, tau); h = 1e-4
            J = np.column_stack([(rk4(fk, c + h * I6[i], tau) - rk4(fk, c - h * I6[i], tau)) / (2 * h) for i in range(D)])
            hh = 2e-3; Hs = {}
            for i in range(D):
                for j in range(i, D):
                    a, b = hh * I6[i], hh * I6[j]
                    Hs[i, j] = Hs[j, i] = (rk4(fk, c + a + b, tau) - rk4(fk, c + a - b, tau) - rk4(fk, c - a + b, tau) + rk4(fk, c - a - b, tau)) / (4 * hh * hh)
            eq = 0.0
            for d in dirs:
                tr = rk4(fk, c + d, tau); quad = P0 + J @ d + 0.5 * sum(Hs[i, j] * d[i] * d[j] for i in range(D) for j in range(D))
                eq = max(eq, np.linalg.norm(tr - quad))
            res[float(tau)] = eq
        ok = [t for t in TAUS if all(res[float(s)] <= tol for s in TAUS if s <= t)]; rows.append(max(ok) if ok else 0.0)
tm = np.array(rows); print('car aniso r_xy=%.1f r_rest=%.2f tol=%g квадр: τ_max 10%%=%.2f медиана=%.2f доля τ=1: %.2f (N=%d)' % (rxy, rr, tol, np.percentile(tm, 10), np.median(tm), np.mean(tm >= 1.0), len(tm)), flush=True)
