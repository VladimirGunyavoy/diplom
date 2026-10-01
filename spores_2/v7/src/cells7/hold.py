"""Клетка цели «спектр + LQR» (research-6 `control_spectrum.md`, PLAN п.0з спектр): у цели управление — непрерывное u ∈ [−umax, umax], не вершины.
Клетка цели хранит для 3 слоёв (−1, 0, +1) образ шага τ из цели и якобиан потока (вариационные уравнения): Φ_u(g+δ, τ) ≈ b_u + J_u δ; образ u — квадратичная интерполяция
по трём слоям. Выбор u на шаге: argmin_u Z(u)ᵀ P Z(u), Z(u) = модельный образ (без rk4), P — Риккати линеаризации в цели. Физические координаты (S._f, S._A)."""
import numpy as np
from scipy.linalg import solve_continuous_are


class HoldCell:
    def __init__(self, S, goal, tau=0.1, R=1.0, Q=None, nrk=20):
        self.S, self.g, self.tau = S, np.array(goal, float), tau; um = S.umax; self.umax = um
        self.b, self.J = {}, {}
        for k in (-1, 0, 1):
            x, J = self.g.copy(), np.eye(2); u = k * um; h = tau / nrk
            for _ in range(nrk):
                def d(x, J): return S._f(x, u), S._A(x, u) @ J
                a1, c1 = d(x, J); a2, c2 = d(x + h / 2 * a1, J + h / 2 * c1); a3, c3 = d(x + h / 2 * a2, J + h / 2 * c2); a4, c4 = d(x + h * a3, J + h * c3)
                x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); J = J + h / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
            self.b[k], self.J[k] = x - self.g, J
        A = S._A(self.g, 0.0); B = ((S._f(self.g, 1e-3) - S._f(self.g, -1e-3)) / 2e-3)[:, None]
        self.P = solve_continuous_are(A, B, np.eye(2) if Q is None else Q, np.array([[R]]))

    def image(self, dx, s):
        """Модельный образ отклонения dx от цели за τ при u = s·umax (квадратичная интерполяция по слоям −1,0,+1)."""
        z = {k: self.b[k] + self.J[k] @ dx for k in (-1, 0, 1)}
        return z[0] + s * (z[1] - z[-1]) / 2 + s * s * ((z[1] + z[-1]) / 2 - z[0])

    def choose(self, dx, mode='spec'):
        cand = {'bb2': (-1.0, 1.0), 'bb3': (-1.0, 0.0, 1.0), 'spec': np.linspace(-1, 1, 401)}[mode]
        cost = [float(self.image(dx, s) @ self.P @ self.image(dx, s)) for s in cand]
        return cand[int(np.argmin(cost))] * self.umax
