"""Системы для клеток v7 (2D): нормированные координаты y = x / scale. Слой k — постоянное управление u_k = ±umax.
Нужны поле f(y,k), якобиан A(y,k) = ∂f/∂y (аналитически), периодические размерности per (в нормированных единицах; 0 — нет)."""
import numpy as np


class Sys2:
    def __init__(self, name, scale, f, A, umax, per=(0.0, 0.0), box=None):
        self.name, self.scale, self._f, self._A, self.umax, self.per = name, np.array(scale, float), f, A, umax, np.array(per, float)
        self.L = 2; self.box = box            # box: ((lo0,hi0),(lo1,hi1)) в нормированных единицах

    def u(self, k): return self.umax * (1.0 if k == 0 else -1.0)
    def f(self, y, k): return self._f(y * self.scale, self.u(k)) / self.scale
    def A(self, y, k): return self._A(y * self.scale, self.u(k)) * self.scale[None, :] / self.scale[:, None]      # S⁻¹ A S
    def wrap(self, d):
        d = np.array(d, float)
        for i in range(2):
            if self.per[i] > 0: d[..., i] = (d[..., i] + self.per[i] / 2) % self.per[i] - self.per[i] / 2
        return d

    def rk4(self, y, k, t, n=None):
        """Точный (мелким rk4) поток нормированной точки/массива y (...,2) за время t."""
        y = np.array(y, float); n = n or max(4, int(np.ceil(abs(t) / 0.02))); h = t / n
        for _ in range(n):
            k1 = self.f(y, k); k2 = self.f(y + h / 2 * k1, k); k3 = self.f(y + h / 2 * k2, k); k4 = self.f(y + h * k3, k); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y


def di(umax=1.0, vmax=4.0):
    """Двойной интегратор: x' = v, v' = u. Поток аффинный по состоянию — модель точна."""
    f = lambda x, u: np.stack([x[..., 1], np.full(x.shape[:-1], u)], -1)
    A = lambda x, u: np.array([[0.0, 1.0], [0.0, 0.0]])
    return Sys2('DI', (4.0, vmax), f, A, umax, box=((-1, 1), (-1, 1)))


def pend(umax=0.3, wmax=4.0):
    """Маятник θ'' = −sin θ + u, θ периодично (в норм. единицах период 2)."""
    f = lambda x, u: np.stack([x[..., 1], -np.sin(x[..., 0]) + u], -1)
    A = lambda x, u: np.array([[0.0, 1.0], [-np.cos(x[..., 0]), 0.0]]) if x.ndim == 1 else None
    def A_(x, u):
        x = np.asarray(x, float)
        if x.ndim == 1: return np.array([[0.0, 1.0], [-np.cos(x[0]), 0.0]])
        M = np.zeros(x.shape[:-1] + (2, 2)); M[..., 0, 1] = 1.0; M[..., 1, 0] = -np.cos(x[..., 0]); return M
    return Sys2('pend', (np.pi, wmax), f, A_, umax, per=(2.0, 0.0), box=((-1, 1), (-1, 1)))
