"""
Pendulum - Normalized Pendulum
===============================

state   = [theta, omega]   theta=0 — нижнее (устойчивое) положение
control = u                нормированный момент/ускорение

Нормировка: g/l = 1, m*l^2 = 1.
    theta_ddot = -sin(theta) + u

Динамика нелинейная — аналитического шага нет, step() интегрирует
RK4 с n_sub подшагами на интервале dt.

a_max is synced from shared_context.param_manager each tick.
step() is stateless: takes (x0, v0, u, dt) and returns (theta, omega).
Works for scalars or numpy arrays (BranchFamily calls it vectorized over rays).
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext


def _step(x0, v0, u, dt, n_sub: int = 10):
    h = dt / n_sub
    theta, omega = x0, v0
    for _ in range(n_sub):
        k1_th = omega
        k1_om = -np.sin(theta) + u

        k2_th = omega + 0.5 * h * k1_om
        k2_om = -np.sin(theta + 0.5 * h * k1_th) + u

        k3_th = omega + 0.5 * h * k2_om
        k3_om = -np.sin(theta + 0.5 * h * k2_th) + u

        k4_th = omega + h * k3_om
        k4_om = -np.sin(theta + h * k3_th) + u

        theta = theta + (h / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega = omega + (h / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega


class Pendulum:
    """
    Normalized pendulum (g/l = 1, m*l^2 = 1).
    state = [theta, omega], theta=0 — нижнее устойчивое положение.
    Stateless step — RK4 с n_sub подшагами (нелинейная динамика).
    a_max читается из shared_context.param_manager и кэшируется через tick().
    """

    def __init__(self, ctx: "SharedContext"):
        self._ctx = ctx
        self.a_max: float = ctx.param_manager.a_max

    def tick(self) -> None:
        self.a_max = self._ctx.param_manager.a_max

    def deriv(self, x, v, u):
        """Phase-space velocity: dtheta/dt = omega, domega/dt = -sin(theta) + u."""
        return v, -np.sin(x) + u

    def step(self, x0, v0, u, dt):
        """Compute one step from (theta0, omega0) under control u for time dt. Returns (theta, omega)."""
        return _step(x0, v0, u, dt)

    def __repr__(self) -> str:
        return f"Pendulum(a_max={self.a_max:.3f})"
