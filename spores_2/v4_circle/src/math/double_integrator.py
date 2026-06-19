"""
DoubleIntegrator - 1D Double Integrator
========================================

state   = [x, v]
control = u  (acceleration)

Exact analytical step (constant u over interval):
    x_new = x + v*dt + 0.5*u*dt^2
    v_new = v + u*dt

a_max is synced from shared_context.param_manager each tick.
step() is stateless: takes (x0, v0, u, dt) and returns (x, v).
Works for scalars or numpy arrays (BranchFamily calls it vectorized over rays).

JIT note:
    _step() is written for numba compatibility.
    To enable: decorate with @njit from numba.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext


def _step(x0, v0, u, dt):
    new_x = x0 + v0 * dt + 0.5 * u * dt * dt
    new_v = v0 + u * dt
    return new_x, new_v


class DoubleIntegrator:
    """
    1D double integrator (x_ddot = u).
    Stateless step — no internal position state.
    a_max is read from shared_context.param_manager and cached via tick().
    """

    def __init__(self, ctx: "SharedContext"):
        self._ctx = ctx
        self.a_max: float = ctx.param_manager.a_max

    def tick(self) -> None:
        self.a_max = self._ctx.param_manager.a_max

    def derivative(self, x, v, u):
        """Phase velocity: f(x, v, u) = (v, u). Returns (dx/dt, dv/dt)."""
        return v, u

    def step(self, x0, v0, u, dt):
        """Compute one step from (x0, v0) under control u for time dt. Returns (x, v)."""
        return _step(x0, v0, u, dt)

    def __repr__(self) -> str:
        return f"DoubleIntegrator(a_max={self.a_max:.3f})"
