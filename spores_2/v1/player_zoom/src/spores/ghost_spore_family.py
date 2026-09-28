"""
GhostSporeFamily - Grid of ghost spores via recursive double integrator evolution
==================================================================================

Grid: i=1..n_tau (generations/time), j=-n_u..n_u (control values)

Evolution:
  state(0, j) = root (ghost_spore_0) for all j
  state(i, j) = di.step(state(i-1, j), u_j, delta_tau)

  delta_tau = tau / n_tau
  u_j       = (j / n_u) * a_max

Graph edges:
  time:    (i,j) -- (i+1,j)
  control: (i,j) -- (i,j+1)

Rebuild (new Entity objects): only when n_tau or n_u changes.
Recompute positions: every tick (cursor move, tau, a_max changes).
"""

import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from .spore import GhostSpore
from ..math.double_integrator import DoubleIntegrator

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext
    from ..core.line_manager import LineManager
    from ..core.color_manager import ColorManager
    from .spore_manager import SporeManager


class _GhostLineFamily:

    def __init__(self, line_manager: "LineManager", color_manager: "ColorManager",
                 name_prefix: str = 'fam', color_key: str = 'family'):
        self._lm = line_manager
        self._cm = color_manager
        self._prefix = name_prefix
        self._color_key = color_key
        self._edge_nodes: Dict[str, Tuple] = {}  # name -> (key1, key2)
        self._generation = 0

    def build(self, nodes: Dict, n_tau: int, n_u: int, root: GhostSpore) -> None:
        for name in self._edge_nodes:
            self._lm.disable(name)
        self._edge_nodes = {}

        c_time = self._cm.get_color(self._color_key, 'edge_time')
        a_time = self._cm.get_rgba(self._color_key, 'edge_time')[3]
        c_ctrl = self._cm.get_color(self._color_key, 'edge_control')
        a_ctrl = self._cm.get_rgba(self._color_key, 'edge_control')[3]
        p = self._prefix
        gen = self._generation
        dummy = np.zeros(3)

        # root -> (1,j)  [None marks root as source]
        for j in range(-n_u, n_u + 1):
            name = f'{p}_g{gen}_re_{j}'
            self._lm.create(name, dummy, dummy, c_time, a_time)
            self._edge_nodes[name] = (None, (1, j))

        # (i,j) -> (i+1,j)
        for i in range(1, n_tau):
            for j in range(-n_u, n_u + 1):
                name = f'{p}_g{gen}_te_{i}_{j}'
                self._lm.create(name, dummy, dummy, c_time, a_time)
                self._edge_nodes[name] = ((i, j), (i + 1, j))

        # (i,j) -> (i,j+1)
        for i in range(1, n_tau + 1):
            for j in range(-n_u, n_u):
                name = f'{p}_g{gen}_ce_{i}_{j}'
                self._lm.create(name, dummy, dummy, c_ctrl, a_ctrl)
                self._edge_nodes[name] = ((i, j), (i, j + 1))

        self._generation += 1
        print(f"[_GhostLineFamily] Built {len(self._edge_nodes)} edges (gen {self._generation})")

    def update(self, nodes: Dict, root: GhostSpore) -> None:
        for name, (k1, k2) in self._edge_nodes.items():
            p1 = root.real_position if k1 is None else nodes[k1].real_position
            self._lm.update(name, p1, nodes[k2].real_position)


class GhostSporeFamily:

    def __init__(self, root: GhostSpore, ctx: "SharedContext",
                 name: str = 'fam', time_sign: int = 1, color_key: str = 'family'):
        self._root = root
        self._ctx = ctx
        self._name = name
        self._time_sign = time_sign
        self._color_key = color_key
        self._spore_manager: "SporeManager" = ctx.spore_manager
        self._di = DoubleIntegrator(ctx)
        self._nodes: Dict[Tuple[int, int], GhostSpore] = {}
        self._n_tau: int = -1
        self._n_u: int = -1
        self._generation: int = 0

        lm = getattr(ctx, 'line_manager', None)
        cm = getattr(ctx, 'color_manager', None)
        self._line_family: Optional[_GhostLineFamily] = (
            _GhostLineFamily(lm, cm, name_prefix=name, color_key=color_key) if lm and cm else None
        )

        self._build()

    def _build(self) -> None:
        for spore in self._nodes.values():
            spore.enabled = False
        self._nodes.clear()

        pm = self._ctx.param_manager
        n_tau = int(pm.n_tau)
        n_u = int(pm.n_u)
        self._n_tau = n_tau
        self._n_u = n_u

        for i in range(1, n_tau + 1):
            for j in range(-n_u, n_u + 1):
                node_name = f'{self._name}_g{self._generation}_{i}_{j}'
                spore = self._spore_manager.create(GhostSpore, node_name)
                self._nodes[(i, j)] = spore

        cm = getattr(self._ctx, 'color_manager', None)
        if cm:
            c_node = cm.get_color(self._color_key, 'node')
            a_node = cm.get_rgba(self._color_key, 'node')[3]
            for spore in self._nodes.values():
                spore.color = c_node
                spore.alpha = a_node

            c_root = cm.get_color(self._color_key, 'root')
            a_root = cm.get_rgba(self._color_key, 'root')[3]
            self._root.color = c_root
            self._root.alpha = a_root

            if n_u > 0:
                c_plus  = cm.get_color('boundary', 'plus_u')
                a_plus  = cm.get_rgba('boundary', 'plus_u')[3]
                c_minus = cm.get_color('boundary', 'minus_u')
                a_minus = cm.get_rgba('boundary', 'minus_u')[3]
                for i in range(1, n_tau + 1):
                    self._nodes[(i,  n_u)].color = c_plus
                    self._nodes[(i,  n_u)].alpha = a_plus
                    self._nodes[(i, -n_u)].color = c_minus
                    self._nodes[(i, -n_u)].alpha = a_minus
        else:
            for spore in self._nodes.values():
                spore.alpha = 0.4

        if self._line_family:
            self._line_family.build(self._nodes, n_tau, n_u, self._root)

        self._generation += 1
        print(f"[GhostSporeFamily] Built {n_tau}x{2*n_u+1} grid (gen {self._generation})")

    def _recompute(self) -> None:
        self._di.tick()
        pm = self._ctx.param_manager
        n_tau = self._n_tau
        n_u = self._n_u
        a_max = self._di.a_max
        tau = pm.tau
        dt = (tau / n_tau if n_tau > 0 else tau) * self._time_sign

        root_pos = self._root.real_position  # [x, y_offset, v]

        for i in range(1, n_tau + 1):
            for j in range(-n_u, n_u + 1):
                u_j = (j / n_u) * a_max if n_u > 0 else 0.0
                parent_pos = root_pos if i == 1 else self._nodes[(i - 1, j)].real_position
                s = self._di.step(x0=parent_pos[0], v0=parent_pos[2], u=u_j, t=dt)
                self._nodes[(i, j)].real_position = np.array([s[0], 0.01, s[1]])

        if self._line_family:
            self._line_family.update(self._nodes, self._root)

    @property
    def nodes(self) -> Dict[Tuple[int, int], GhostSpore]:
        return self._nodes

    @property
    def n_tau(self) -> int:
        return self._n_tau

    @property
    def n_u(self) -> int:
        return self._n_u

    @property
    def time_sign(self) -> int:
        return self._time_sign

    def tick(self) -> None:
        pm = self._ctx.param_manager
        n_tau = int(pm.n_tau)
        n_u = int(pm.n_u)

        if n_tau != self._n_tau or n_u != self._n_u:
            self._build()

        if n_tau == 0:
            return

        self._recompute()
