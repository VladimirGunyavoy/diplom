"""Клетка цели «спектр + LQR» в nD (обобщение hold.py): x' = f(x,u), u ∈ [−1,1]^m (физические, умноженные на scale), цель x*, равновесное u_eq (f(x*,u_eq)=0, наименьшие квадраты).
Слои по каналу: u_i ∈ {−1, u_eq_i, +1} (остальные каналы = u_eq) — 2m+1 образов шага τ и якобианов потока (конечные разности по x); образ u — разделимая квадратичная интерполяция Лагранжа
по каналам (перекрёстные члены 2-го порядка не учитываются — у манипулятора они малы около цели). u = argmin Z(u)ᵀPZ(u) на коробке (L-BFGS-B, старт u_eq / прошлое u). P — Риккати линеаризации."""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize


def rk4(f, x, u, t, n=20):
    h = t / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class HoldCellN:
    def __init__(self, f, goal, m, tau=0.1, Q=None, R=1.0, eps=1e-6, nrk=20):
        self.f, self.g, self.m, self.tau, self.n = f, np.array(goal, float), m, tau, len(goal); n = self.n
        from scipy.optimize import least_squares
        self.ueq = np.clip(least_squares(lambda u: f(self.g, u), np.zeros(m)).x, -1, 1)
        def jac(fun, x):
            return np.stack([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(len(x))], -1)
        A = jac(lambda x: f(x, self.ueq), self.g); B = jac(lambda u: f(self.g, u), self.ueq)
        self.P = solve_continuous_are(A, B, np.eye(n) if Q is None else Q, R * np.eye(m))
        self.layers = {}                                           # (i, side) → (b, J): i — канал (−1 — центр), side ∈ {−1,0,+1}
        def layer(u):
            fl = lambda x: rk4(f, x, u, tau, nrk); return fl(self.g) - self.g, jac(fl, self.g)
        self.layers[(-1, 0)] = layer(self.ueq)
        for i in range(m):
            for s in (-1, 1):
                u = self.ueq.copy(); u[i] = s; self.layers[(i, s)] = layer(u)

    def _basis(self, a, c):
        """Лагранж по узлам (−1, c, +1) в точке a."""
        lo, hi = -1.0, 1.0
        if hi - c < 1e-9 or c - lo < 1e-9: return np.array([0.0, 1.0, 0.0]) if abs(a - c) < 1e-12 else np.array([(a == lo) * 1.0, 0.0, (a == hi) * 1.0])
        return np.array([(a - c) * (a - hi) / ((lo - c) * (lo - hi)), (a - lo) * (a - hi) / ((c - lo) * (c - hi)), (a - lo) * (a - c) / ((hi - lo) * (hi - c))])

    def image(self, dx, u):
        b0, J0 = self.layers[(-1, 0)]; Z0 = b0 + J0 @ dx; Z = Z0.copy()
        for i in range(self.m):
            w = self._basis(u[i], self.ueq[i]); zs = []
            for s in (-1, 1): b, J = self.layers[(i, s)]; zs.append(b + J @ dx)
            Z = Z + w[0] * (zs[0] - Z0) + w[2] * (zs[1] - Z0)
        return Z

    def choose(self, dx, u0=None):
        cost = lambda u: float((lambda z: z @ self.P @ z)(self.image(dx, u)))
        r = minimize(cost, self.ueq if u0 is None else u0, bounds=[(-1, 1)] * self.m, method='L-BFGS-B'); return r.x

    def choose_vertices(self, dx):
        import itertools
        best = min(itertools.product((-1.0, 1.0), repeat=self.m), key=lambda u: (lambda z: z @ self.P @ z)(self.image(dx, np.array(u)))); return np.array(best)
