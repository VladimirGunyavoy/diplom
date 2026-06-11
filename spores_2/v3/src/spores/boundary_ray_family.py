"""
BranchFamily - Bang-bang control trajectories from root spore
=============================================================

Template: ((u1_sign, t1_sign), (u2_sign, t2_sign))
  u1_sign, u2_sign : control signs (+1/-1), must be opposite
  t1_sign, t2_sign : time signs (+1/-1), can be equal

For each switch moment k in 1..n_tau:
  steps 1..k       : control = u1_sign * a_max,  dt = dtau * t1_sign
  steps k+1..n_tau : control = u2_sign * a_max,  dt = dtau * t2_sign
  k == n_tau means no switch (full phase-1 trajectory)

Produces n_tau rays, each with n_tau nodes.
Rebuild (new Entity objects): when n_tau changes.
Recompute positions: every tick.
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from .spore import GhostSpore, Spore

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext
    from ..core.line_manager import LineManager
    from ..core.color_manager import ColorManager
    from ..core.spore_manager import SporeManager
    from ..core.surface_manager import SurfaceManager


class _BranchRay:

    def __init__(self, name: str, n_tau: int, switch_k: int,
                 spore_manager: "SporeManager", line_manager: "LineManager",
                 c_node, a_node,
                 c_p1, a_p1, t1_sign: int,
                 c_p2, a_p2, t2_sign: int):
        self._lm = line_manager
        self._nodes: List[Spore] = []
        self._line_names: List[str] = []
        dummy = np.zeros(3)

        for k in range(n_tau):
            spore = spore_manager.create(Spore, f'{name}_n{k}')
            spore.color = c_node
            spore.alpha = a_node
            self._nodes.append(spore)

        for j in range(n_tau - 1):
            lname = f'{name}_l{j}'
            # Line j covers step j+2: phase 1 if step j+2 <= switch_k
            if j < switch_k - 1:
                c, a, t = c_p1, a_p1, t1_sign
            else:
                c, a, t = c_p2, a_p2, t2_sign
            line_manager.create_arrow(lname, dummy, dummy, c, a, t)
            self._line_names.append(lname)

    def update(self, positions: np.ndarray) -> None:
        for k, pos in enumerate(positions):
            self._nodes[k].real_position = pos
        for k, lname in enumerate(self._line_names):
            self._lm.update(lname, positions[k], positions[k + 1])

    def disable(self) -> None:
        for spore in self._nodes:
            spore.enabled = False
        for lname in self._line_names:
            self._lm.disable(lname)

    def set_visible(self, visible: bool) -> None:
        for spore in self._nodes:
            spore.enabled = visible
        for lname in self._line_names:
            self._lm.set_visible(lname, visible)


class BranchFamily:

    def __init__(self, root: "GhostSpore", template: Tuple, ctx: "SharedContext",
                 color_key: str = 'ray', color_suffix: str = 'plus_u', name: str = 'branch'):
        assert template[0][0] != template[1][0], "u1 and u2 must have opposite signs"

        self._root = root
        self._template = template
        self._ctx = ctx
        self._color_key = color_key
        self._color_suffix = color_suffix
        self._name = name

        self._sm: "SporeManager" = ctx.spore_manager
        self._lm: "LineManager" = ctx.line_manager
        self._cm: "ColorManager" = ctx.color_manager
        self._sfm: "SurfaceManager" = ctx.surface_manager

        self._rays: List[_BranchRay] = []
        self._root_line: str = ''
        self._root_line2: str = ''
        self._envelope_lines: List[str] = []
        self._grid_lines: List[str] = []
        self._grid_index: List[Tuple[int, int]] = []
        self._fan_surface: str = ''
        self._visible: bool = True
        self._n_tau: int = -1
        self._generation: int = 0

        self._cached_root: np.ndarray = np.full(3, np.nan)
        self._cached_a_max: float = np.nan
        self._cached_tau: float = np.nan

        self._build()

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        for ray in self._rays:
            ray.set_visible(visible)
        for lname in [self._root_line, self._root_line2] + self._envelope_lines + self._grid_lines:
            if lname:
                self._lm.set_visible(lname, visible)
        if self._fan_surface:
            self._sfm.set_visible(self._fan_surface, visible)

    def toggle(self) -> None:
        self.set_visible(not self._visible)

    def _phase_edge(self, u_sign: int, t_sign: int):
        t_key = 'fwd' if t_sign > 0 else 'bwd'
        u_key = 'pos' if u_sign > 0 else 'neg'
        key = f'edge_{t_key}_{u_key}'
        c = self._cm.get_color('ray', key)
        a = self._cm.get_rgba('ray', key)[3]
        return c, a

    def _build(self) -> None:
        for ray in self._rays:
            ray.disable()
        self._rays = []
        for lname in [self._root_line, self._root_line2] + self._envelope_lines + self._grid_lines:
            if lname:
                self._lm.disable(lname)
        if self._fan_surface:
            self._sfm.disable(self._fan_surface)
        self._envelope_lines = []
        self._grid_lines = []
        self._grid_index = []
        self._fan_surface = ''

        pm = self._ctx.param_manager
        n_tau = int(pm.n_tau)
        self._n_tau = n_tau

        if n_tau == 0:
            return

        c_node = self._cm.get_color(self._color_key, f'node_{self._color_suffix}')
        a_node = self._cm.get_rgba(self._color_key, f'node_{self._color_suffix}')[3]

        (u1_sign, t1_sign), (u2_sign, t2_sign) = self._template
        c_p1, a_p1 = self._phase_edge(u1_sign, t1_sign)
        c_p2, a_p2 = self._phase_edge(u2_sign, t2_sign)
        c_edge = self._cm.get_color(self._color_key, f'edge_{self._color_suffix}')
        a_edge = self._cm.get_rgba(self._color_key, f'edge_{self._color_suffix}')[3]

        gen = self._generation
        dummy = np.zeros(3)

        # k=0..n_tau: k=0 is pure phase-2, k=n_tau is pure phase-1
        for k in range(0, n_tau + 1):
            name = f'{self._name}_g{gen}_k{k}'
            ray = _BranchRay(name, n_tau, k, self._sm, self._lm,
                             c_node, a_node,
                             c_p1, a_p1, t1_sign,
                             c_p2, a_p2, t2_sign)
            self._rays.append(ray)

        # root lines represent real step-1 evolution
        # ray 0 (pure phase-2): step 1 uses phase-2 params
        # ray n_tau (pure phase-1): step 1 uses phase-1 params
        self._root_line = f'{self._name}_g{gen}_root0'
        self._lm.create_arrow(self._root_line, dummy, dummy, c_p2, a_p2, t2_sign)
        self._root_line2 = f'{self._name}_g{gen}_root_ntau'
        self._lm.create_arrow(self._root_line2, dummy, dummy, c_p1, a_p1, t1_sign)

        for i in range(n_tau):
            lname = f'{self._name}_g{gen}_env{i}'
            self._lm.create(lname, dummy, dummy, c_edge, a_edge)
            self._envelope_lines.append(lname)


        c_surf = self._cm.get_color(self._color_key, f'surface_{self._color_suffix}')
        a_surf = self._cm.get_rgba(self._color_key, f'surface_{self._color_suffix}')[3]
        self._fan_surface = f'{self._name}_g{gen}_fan'
        self._sfm.create_fan(self._fan_surface, n_tau=n_tau, color=c_surf, alpha=a_surf)

        self._generation += 1
        print(f"[BranchFamily] Built {len(self._rays)} rays (gen {self._generation})")

        self.set_visible(self._visible)

    def _recompute(self) -> None:
        pm = self._ctx.param_manager
        n_tau = self._n_tau
        a_max = pm.a_max
        dtau = pm.tau / n_tau

        (u1_sign, t1_sign), (u2_sign, t2_sign) = self._template
        u1 = u1_sign * a_max
        u2 = u2_sign * a_max
        dt1 = dtau * t1_sign
        dt2 = dtau * t2_sign

        root_pos = self._root.real_position  # [x, y_offset, v]
        x0, v0 = root_pos[0], root_pos[2]

        # k=0..n_tau: n_tau+1 rays
        x = np.full(n_tau + 1, x0)
        v = np.full(n_tau + 1, v0)
        k_arr = np.arange(0, n_tau + 1)   # switch moments: 0=pure phase-2, n_tau=pure phase-1

        positions = np.zeros((n_tau + 1, n_tau, 3))  # [ray, step, xyz]

        for step_i in range(1, n_tau + 1):
            mask = step_i <= k_arr
            u  = np.where(mask, u1, u2)
            dt = np.where(mask, dt1, dt2)
            x, v = self._ctx.model.step(x, v, u, dt)
            positions[:, step_i - 1, 0] = x
            positions[:, step_i - 1, 1] = 0.03
            positions[:, step_i - 1, 2] = v

        for ray_idx, ray in enumerate(self._rays):
            ray.update(positions[ray_idx])

        self._lm.update(self._root_line, root_pos, positions[0, 0])
        self._lm.update(self._root_line2, root_pos, positions[-1, 0])

        endpoints = positions[:, -1, :]  # (n_tau, 3)
        for i, lname in enumerate(self._envelope_lines):
            self._lm.update(lname, endpoints[i], endpoints[i + 1])

        for lname, (k_idx, j) in zip(self._grid_lines, self._grid_index):
            self._lm.update(lname, positions[k_idx, k_idx + j - 1], positions[k_idx + 1, k_idx + j])

        surf_positions = positions.copy()
        surf_positions[:, :, 1] = 0.01
        self._sfm.update_fan(self._fan_surface, root_pos, surf_positions)

    def tick(self) -> None:
        pm = self._ctx.param_manager
        n_tau = int(pm.n_tau)

        if n_tau != self._n_tau:
            self._build()
            self._cached_root = np.full(3, np.nan)  # сбрасываем кеш после rebuild

        if self._n_tau == 0:
            return

        root_pos = self._root.real_position
        a_max = pm.a_max
        tau = pm.tau

        dirty = (
            not np.array_equal(root_pos, self._cached_root)
            or a_max != self._cached_a_max
            or tau != self._cached_tau
        )
        if not dirty:
            return

        self._cached_root = root_pos.copy()
        self._cached_a_max = a_max
        self._cached_tau = tau

        self._recompute()
