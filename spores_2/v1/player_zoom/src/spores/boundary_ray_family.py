"""
BoundaryRayFamily - Second-generation trajectories from family boundary
========================================================================

For each boundary point (i, ±n_u) of a GhostSporeFamily:
  - Remaining steps: n_tau - i
  - Control: opposite to boundary side (-u_max for +n_u, +u_max for -n_u)
  - Time direction: same as parent family

Boundary sides:
  'plus_u'  — j = +n_u, rays use -u_max
  'minus_u' — j = -n_u, rays use +u_max
"""

import numpy as np
from typing import List, TYPE_CHECKING
from .spore import GhostSpore
from ..math.double_integrator import DoubleIntegrator

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext
    from ..core.line_manager import LineManager
    from ..core.color_manager import ColorManager
    from ..core.spore_manager import SporeManager
    from .ghost_spore_family import GhostSporeFamily


class BoundaryRay:

    def __init__(self, name: str, steps: int, side: str,
                 spore_manager: "SporeManager", line_manager: "LineManager",
                 color_manager: "ColorManager"):
        self._name = name
        self._steps = steps
        self._lm = line_manager
        self._nodes: List[GhostSpore] = []
        self._line_names: List[str] = []

        c_node = color_manager.get_color('ray', f'node_{side}')
        a_node = color_manager.get_rgba('ray', f'node_{side}')[3]
        c_edge = color_manager.get_color('ray', f'edge_{side}')
        a_edge = color_manager.get_rgba('ray', f'edge_{side}')[3]
        dummy = np.zeros(3)

        for k in range(steps):
            spore = spore_manager.create(GhostSpore, f'{name}_n{k}')
            spore.color = c_node
            spore.alpha = a_node
            self._nodes.append(spore)

        for k in range(steps):
            lname = f'{name}_l{k}'
            line_manager.create(lname, dummy, dummy, c_edge, a_edge)
            self._line_names.append(lname)

    def recompute(self, start_pos: np.ndarray, u: float, dt: float, di: DoubleIntegrator) -> None:
        prev = start_pos
        for k in range(self._steps):
            s = di.step(x0=prev[0], v0=prev[2], u=u, t=dt)
            pos = np.array([s[0], 0.01, s[1]])
            self._nodes[k].real_position = pos
            self._lm.update(self._line_names[k], prev, pos)
            prev = pos

    def disable(self) -> None:
        for spore in self._nodes:
            spore.enabled = False
        for lname in self._line_names:
            self._lm.disable(lname)


class BoundaryRayFamily:

    def __init__(self, family: "GhostSporeFamily", side: str, ctx: "SharedContext"):
        self._family = family
        self._side = side      # 'plus_u' or 'minus_u'
        self._ctx = ctx
        self._di = DoubleIntegrator(ctx)
        self._rays: List[BoundaryRay] = []
        self._n_tau: int = -1
        self._n_u: int = -1
        self._generation: int = 0

        self._sm: "SporeManager" = ctx.spore_manager
        self._lm: "LineManager" = ctx.line_manager
        self._cm: "ColorManager" = ctx.color_manager

        self._build()

    def _build(self) -> None:
        for ray in self._rays:
            ray.disable()
        self._rays = []

        n_tau = self._family.n_tau
        n_u = self._family.n_u
        self._n_tau = n_tau
        self._n_u = n_u

        if n_tau == 0 or n_u == 0:
            return

        j = n_u if self._side == 'plus_u' else -n_u
        gen = self._generation

        for i in range(1, n_tau):  # last point has 0 steps remaining
            steps = n_tau - i
            name = f'ray_{self._side}_g{gen}_{i}'
            ray = BoundaryRay(name, steps, self._side, self._sm, self._lm, self._cm)
            self._rays.append(ray)

        self._generation += 1
        print(f"[BoundaryRayFamily] Built {len(self._rays)} rays from {self._side} (gen {self._generation})")

    def _recompute(self) -> None:
        self._di.tick()
        pm = self._ctx.param_manager
        a_max = self._di.a_max
        dt = (pm.tau / self._n_tau) * self._family.time_sign

        j = self._n_u if self._side == 'plus_u' else -self._n_u
        u = -a_max if self._side == 'plus_u' else a_max

        for idx, ray in enumerate(self._rays):
            i = idx + 1
            start_pos = self._family.nodes[(i, j)].real_position
            ray.recompute(start_pos, u, dt, self._di)

    def tick(self) -> None:
        n_tau = self._family.n_tau
        n_u = self._family.n_u

        if n_tau != self._n_tau or n_u != self._n_u:
            self._build()

        if self._n_tau == 0 or self._n_u == 0:
            return

        self._recompute()
