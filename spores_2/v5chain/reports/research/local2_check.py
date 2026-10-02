"""research hub-v5chain-research-8: точность локальной модели 2-го порядка по t и u (идея пользователя 2026-10-02: «разложить следующее состояние
до 2-го порядка по времени и управлению»). Φ2(x0,t,u) = x0 + t·f(x0,u) + t²/2·f_x(x0,u)·f(x0,u); u постоянно на [0,t]. Ошибка против rk4 мелким шагом."""
import numpy as np
def rk(f, x, u, T, n=400):
    h = T / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
def jac(f, x, u, e=1e-6): return np.stack([(f(x + e * np.eye(len(x))[i], u) - f(x - e * np.eye(len(x))[i], u)) / (2 * e) for i in range(len(x))], 1)
def phi2(f, x, u, t): F = f(x, u); return x + t * F + t * t / 2 * jac(f, x, u) @ F
pend = lambda x, u: np.array([x[1], np.sin(x[0]) + u[0]])                          # |u| ≤ .3
dd = lambda x, u: np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])       # v, ω ∈ [−1,1]
rng = np.random.default_rng(0)
for nm, f, X, U in (('маятник', pend, lambda: np.r_[rng.uniform(-np.pi, np.pi), rng.uniform(-2, 2)], lambda: np.r_[rng.choice([-.3, .3])]),
                    ('дифдрайв', dd, lambda: np.r_[0, 0, rng.uniform(-np.pi, np.pi)], lambda: rng.choice([-1., 1.], 2))):
    for t in (.1, .2, .3, .5, 1.):
        E = []
        for _ in range(200):
            x, u = X(), U(); ex = rk(f, x, u, t); E.append(np.linalg.norm(phi2(f, x, u, t) - ex) / max(np.linalg.norm(ex - x), 1e-9))
        print(f'{nm:9s} t={t:.1f}: относит. ошибка (к смещению) med {np.median(E):.1e} p95 {np.quantile(E, .95):.1e}')
