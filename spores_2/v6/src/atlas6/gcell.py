"""Общая клетка v6 для любой системы ẋ = f(x, u) (n=2: маятник и др.): поток — RK4 с шагом ≤ dt_max; сегмент — перпендикуляр к полю
в нормированных координатах (S = diag(Lx, Lv)); стены, вход/выход — интегрированием краёв. Для DI совпадает с аналитической `Cell` (тест)."""
import numpy as np


def rk4(f, x, u, t, dt_max=0.01):
    """Поток x на время t (t<0 — назад), x: (..., 2)."""
    x = np.array(x, float)
    n = max(1, int(np.ceil(abs(t) / dt_max))); h = t / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def pendulum(g_over_l=1.0):
    """θ̈ = −(g/l) sin θ + u: состояние (θ, ω). u = ±umax — слои."""
    return lambda x, u: np.stack([x[..., 1], -g_over_l * np.sin(x[..., 0]) + u], axis=-1)


def double_integrator():
    return lambda x, u: np.stack([x[..., 1], np.full(x.shape[:-1], float(u))], axis=-1)


class GCell:
    def __init__(self, f, center, u, r, tau, Lx=1.0, Lv=1.0, dt_max=0.01):
        self.f, self.c, self.u, self.r, self.tau = f, np.asarray(center, float), float(u), float(r), float(tau)
        self.Lx, self.Lv, self.dt_max = float(Lx), float(Lv), dt_max
        F = f(self.c, self.u); Fn = np.array([F[0] / Lx, F[1] / Lv])
        nn = np.array([-Fn[1], Fn[0]]) / np.hypot(*Fn)
        self.d = np.array([nn[0] * Lx, nn[1] * Lv])

    def flow(self, p, t):
        return rk4(self.f, p, self.u, t, self.dt_max)

    def segment(self, s):
        return self.c + np.asarray(s, float)[..., None] * self.d

    def wall(self, sign, t):
        p = self.segment(sign * self.r)
        return np.array([self.flow(p, ti) for ti in np.atleast_1d(t)])

    def entry(self):
        return np.array([self.flow(self.segment(s), -self.tau) for s in (-self.r, self.r)])

    def exit(self):
        return np.array([self.flow(self.segment(s), self.tau) for s in (-self.r, self.r)])

    def stretch(self, t=None):
        """ρ = нормированная длина торца после потока t / длина сегмента (нормированная)."""
        t = self.tau if t is None else t
        nl = lambda e: float(np.hypot((e[1] - e[0])[0] / self.Lx, (e[1] - e[0])[1] / self.Lv))
        s = self.segment(np.array([-self.r, self.r])); e = np.array([self.flow(p, t) for p in s])
        return nl(e) / nl(s)
