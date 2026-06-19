"""
CirclePattern - Circle of spores with derivative arrows and trajectories
=========================================================================

Places n_circle spores on a circle of given radius around look_point.
For each control (u = -a_max, 0, +a_max):
  - derivative arrow (ScalableTipArrow)
  - trajectory: n_tau segments integrated over time tau
Each control group togglable via toggle_group().
"""

import numpy as np
from typing import List, TYPE_CHECKING

from .spore import Spore

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext
    from ..core.spore_manager import SporeManager
    from ..core.line_manager import LineManager
    from ..core.color_manager import ColorManager


class _ControlGroup:
    def __init__(self, u_sign: int, color, name_prefix: str):
        self.u_sign = u_sign
        self.color = color
        self.prefix = name_prefix
        self.arrow_names: List[str] = []
        self.traj_names: List[List[str]] = []
        self.visible: bool = True


class CirclePattern:

    Y_NODE = 0.03
    Y_EDGE = 0.02
    Y_ARROW = 0.025
    Y_TRAJ = 0.015

    def __init__(self, name: str, shared_context: "SharedContext",
                 spore_manager: "SporeManager", line_manager: "LineManager",
                 color_manager: "ColorManager"):
        self._name = name
        self._ctx = shared_context
        self._spore_manager = spore_manager
        self._line_manager = line_manager
        self._node_color = color_manager.get_color('circle', 'node')
        self._edge_color = color_manager.get_color('circle', 'edge')

        self._groups = [
            _ControlGroup(-1, color_manager.get_color('circle', 'arrow_neg'),  f'{name}_un'),
            _ControlGroup( 0, color_manager.get_color('circle', 'arrow_zero'), f'{name}_u0'),
            _ControlGroup(+1, color_manager.get_color('circle', 'arrow_pos'),  f'{name}_up'),
        ]

        self._spores: List[Spore] = []
        self._edge_names: List[str] = []
        self._n_built = 0
        self._n_tau_built = 0

    def toggle_group(self, index: int) -> None:
        g = self._groups[index]
        g.visible = not g.visible
        for ln in g.arrow_names:
            self._line_manager.set_visible(ln, g.visible)
        for segs in g.traj_names:
            for ln in segs:
                self._line_manager.set_visible(ln, g.visible)
        label = {-1: 'u-', 0: 'u0', 1: 'u+'}[g.u_sign]
        print(f"[CirclePattern] {label} {'ON' if g.visible else 'OFF'}")

    def tick(self) -> None:
        n = int(self._ctx.param_manager.n_circle)
        n_tau = int(self._ctx.param_manager.n_tau)
        if n != self._n_built or n_tau != self._n_tau_built:
            self._rebuild(n, n_tau)

        if self._n_built == 0:
            return

        radius = self._ctx.param_manager.radius
        a_max = self._ctx.param_manager.a_max
        tau = self._ctx.param_manager.tau
        model = self._ctx.model
        lp = self._ctx.look_point

        positions = self._compute_positions(lp, radius, self._n_built)

        for i, spore in enumerate(self._spores):
            spore.real_position = positions[i]

        for i, ln in enumerate(self._edge_names):
            j = (i + 1) % self._n_built
            self._line_manager.update(ln, positions[i], positions[j])

        dt = tau / self._n_tau_built if self._n_tau_built > 0 else 0.0

        for g in self._groups:
            if not g.visible:
                continue
            u_val = g.u_sign * a_max
            for i in range(self._n_built):
                x = positions[i][0]
                z = positions[i][2]

                # derivative arrow
                dx, dz = model.derivative(x, z, u_val)
                norm = np.sqrt(dx * dx + dz * dz)
                if norm < 1e-12:
                    self._line_manager.update(g.arrow_names[i], positions[i], positions[i])
                else:
                    arrow_len = (radius / 3.0) * np.tanh(3.0 / radius * norm) if radius > 1e-8 else 0.0
                    adx, adz = dx / norm * arrow_len, dz / norm * arrow_len
                    p2 = np.array([x + adx, self.Y_ARROW, z + adz], dtype=float)
                    self._line_manager.update(g.arrow_names[i], positions[i], p2)

                # trajectory
                if self._n_tau_built == 0:
                    continue
                tx, tz = x, z
                for j, ln in enumerate(g.traj_names[i]):
                    p1 = np.array([tx, self.Y_TRAJ, tz], dtype=float)
                    tx, tz = model.step(tx, tz, u_val, dt)
                    p2 = np.array([tx, self.Y_TRAJ, tz], dtype=float)
                    self._line_manager.update(ln, p1, p2)

    def _compute_positions(self, lp, radius, n):
        positions = []
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            x = lp[0] + radius * np.cos(theta)
            z = lp[1] + radius * np.sin(theta)
            positions.append(np.array([x, self.Y_NODE, z], dtype=float))
        return positions

    def _rebuild(self, n: int, n_tau: int) -> None:
        for spore in self._spores:
            spore.disable()
            spore.visible = False
        self._spores.clear()

        for ln in self._edge_names:
            self._line_manager.disable(ln)
        self._edge_names.clear()

        for g in self._groups:
            for ln in g.arrow_names:
                self._line_manager.disable(ln)
            g.arrow_names.clear()
            for segs in g.traj_names:
                for ln in segs:
                    self._line_manager.disable(ln)
            g.traj_names.clear()

        p0 = np.array([0, self.Y_NODE, 0], dtype=float)

        for i in range(n):
            spore = self._spore_manager.create(Spore, f'{self._name}_n{i}')
            spore.color = self._node_color
            self._spores.append(spore)

        for i in range(n):
            ln_name = f'{self._name}_e{i}'
            self._line_manager.create(ln_name, p0, p0, self._edge_color, alpha=0.7)
            self._edge_names.append(ln_name)

        for g in self._groups:
            for i in range(n):
                ln_name = f'{g.prefix}_a{i}'
                self._line_manager.create_tip_arrow(ln_name, p0, p0, g.color)
                g.arrow_names.append(ln_name)

                segs = []
                for j in range(n_tau):
                    seg_name = f'{g.prefix}_t{i}_{j}'
                    self._line_manager.create(seg_name, p0, p0, g.color, alpha=0.6)
                    segs.append(seg_name)
                g.traj_names.append(segs)

                if not g.visible:
                    self._line_manager.set_visible(ln_name, False)
                    for seg_name in segs:
                        self._line_manager.set_visible(seg_name, False)

        self._n_built = n
        self._n_tau_built = n_tau
