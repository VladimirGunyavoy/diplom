"""
OrthoGrid - Flow-orthogonal grid in phase space
================================================

Builds a grid aligned with the flow for a fixed control u:
- Time axis: integrate look_point forward/backward n_tau steps
- Ortho axis: step orthogonal to phase velocity, then integrate

Phase-space coordinates (x, v) map to Ursina as (x, y_offset, v).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.shared_context import SharedContext
    from ..core.line_manager import LineManager
    from ..core.surface_manager import SurfaceManager
    from ..core.color_manager import ColorManager
    from ..core.spore_manager import SporeManager
    from .spore import Spore


Y_NODES = 0.03


class OrthoGrid:

    def __init__(self, ctx: "SharedContext", u_sign: int = +1,
                 name: str = 'ort'):
        self._ctx = ctx
        self._u_sign = u_sign
        self._name = name

        self._sm: "SporeManager" = ctx.spore_manager
        self._lm: "LineManager" = ctx.line_manager
        self._sfm: "SurfaceManager" = ctx.surface_manager
        self._cm: "ColorManager" = ctx.color_manager

        self._active: bool = True
        self._mode: int = 0  # 0=time, 1=arc, 2=ortho_front
        self._MODE_NAMES = ['TIME', 'ARC', 'ORTHO']

        self._traj_nodes: List[List["Spore"]] = []   # [ort_idx][time_idx]
        self._traj_lines: List[List[str]] = []       # [ort_idx][line_idx]
        self._front_lines: List[List[str]] = []       # [time_idx][pair_idx]
        self._surf_fwd: str = ''
        self._surf_bwd: str = ''

        self._n_tau: int = -1
        self._n_s: int = -1
        self._generation: int = 0

        self._cached_root = np.full(2, np.nan)
        self._cached_a_max: float = np.nan
        self._cached_tau: float = np.nan
        self._cached_r_s: float = np.nan

        self._build()

    def _edge_color(self, t_sign: int):
        t_key = 'fwd' if t_sign > 0 else 'bwd'
        u_key = 'pos' if self._u_sign > 0 else 'neg'
        key = f'edge_{t_key}_{u_key}'
        c = self._cm.get_color('ray', key)
        a = self._cm.get_rgba('ray', key)[3]
        return c, a

    def _build(self) -> None:
        from .spore import Spore

        for nodes in self._traj_nodes:
            for s in nodes:
                s.enabled = False
        for lines in self._traj_lines:
            for ln in lines:
                self._lm.disable(ln)
        for front in self._front_lines:
            for ln in front:
                self._lm.disable(ln)
        for sn in [self._surf_fwd, self._surf_bwd]:
            if sn:
                self._sfm.disable(sn)
        self._traj_nodes = []
        self._traj_lines = []
        self._front_lines = []
        self._surf_fwd = ''
        self._surf_bwd = ''

        pm = self._ctx.param_manager
        n = int(pm.n_tau)
        ns = int(pm.n_s)
        self._n_tau = n
        self._n_s = ns
        if n == 0:
            return

        gen = self._generation
        c_fwd, a_fwd = self._edge_color(+1)
        c_bwd, a_bwd = self._edge_color(-1)
        dummy = np.zeros(3)
        width = 2 * ns + 1  # number of trajectories

        for oi in range(width):
            dist = abs(oi - ns)
            alpha_fade = max(0.3, 1.0 - dist * 0.12) if ns > 0 else 1.0
            nodes = []
            lines = []

            for ti in range(2 * n + 1):
                s = self._sm.create(Spore, f'{self._name}_g{gen}_r{oi}_t{ti}')
                s.color = c_fwd if ti >= n else c_bwd
                s.alpha = alpha_fade
                nodes.append(s)

            for ti in range(2 * n):
                ln = f'{self._name}_g{gen}_r{oi}_tl{ti}'
                if ti < n:
                    self._lm.create_arrow(ln, dummy, dummy, c_bwd, a_bwd * alpha_fade, t_sign=+1)
                else:
                    self._lm.create_arrow(ln, dummy, dummy, c_fwd, a_fwd * alpha_fade, t_sign=+1)
                lines.append(ln)

            self._traj_nodes.append(nodes)
            self._traj_lines.append(lines)

        # front lines at every time step between adjacent trajectories
        for ti in range(2 * n + 1):
            is_end = (ti == 0 or ti == 2 * n)
            is_center = (ti == n)
            alpha = 0.85 if is_end else (0.4 if is_center else 0.2)
            c = c_bwd if ti < n else c_fwd
            row = []
            for i in range(width - 1):
                ln = f'{self._name}_g{gen}_fr{ti}_{i}'
                self._lm.create(ln, dummy, dummy, c, alpha)
                row.append(ln)
            self._front_lines.append(row)

        # surfaces: fwd and bwd halves, boundary = outermost trajectories + end front
        if ns > 0:
            self._surf_fwd = f'{self._name}_g{gen}_sfwd'
            self._sfm.create_grid(self._surf_fwd, width, n + 1, c_fwd, alpha=0.25)

            self._surf_bwd = f'{self._name}_g{gen}_sbwd'
            self._sfm.create_grid(self._surf_bwd, width, n + 1, c_bwd, alpha=0.25)

        self._generation += 1
        print(f"[OrthoGrid] Built grid {width}x{2*n+1} (gen={self._generation})")

    def cycle_mode(self):
        self._mode = (self._mode + 1) % 3
        self._cached_root = np.full(2, np.nan)
        print(f"[OrthoGrid] Mode: {self._MODE_NAMES[self._mode]}")

    def set_active(self, active: bool):
        self._active = active
        for nodes in self._traj_nodes:
            for s in nodes:
                s.enabled = active
        for lines in self._traj_lines:
            for ln in lines:
                self._lm.set_visible(ln, active)
        for front in self._front_lines:
            for ln in front:
                self._lm.set_visible(ln, active)
        for sn in [self._surf_fwd, self._surf_bwd]:
            if sn:
                self._sfm.set_visible(sn, active)
        if active:
            self._cached_root = np.full(2, np.nan)

    def _ortho_step(self, x, v, u, ds, sign):
        """Step orthogonal to phase velocity. sign=+1 or -1."""
        model = self._ctx.model
        fx, fv = model.deriv(x, v, u)
        norm = np.sqrt(fx * fx + fv * fv)
        if norm < 1e-10:
            ox, ov = 0.0, sign * ds
        else:
            ox = -fv / norm * ds * sign
            ov =  fx / norm * ds * sign
        return x + ox, v + ov

    def _integrate_trajectory(self, x0, v0, u, dtau, n, nodes, lines):
        model = self._ctx.model
        pts = np.zeros((2 * n + 1, 3))
        pts[n] = [x0, Y_NODES, v0]

        x, v = x0, v0
        for j in range(n):
            x, v = model.step(x, v, u, dtau)
            pts[n + j + 1] = [x, Y_NODES, v]

        x, v = x0, v0
        for j in range(n):
            x, v = model.step(x, v, u, -dtau)
            pts[n - j - 1] = [x, Y_NODES, v]

        for i, s in enumerate(nodes):
            s.real_position = pts[i]
        for i, ln in enumerate(lines):
            self._lm.update(ln, pts[i], pts[i + 1])

        return pts

    def _update_visual(self, oi, pts):
        for i, s in enumerate(self._traj_nodes[oi]):
            s.real_position = pts[i]
        for i, ln in enumerate(self._traj_lines[oi]):
            self._lm.update(ln, pts[i], pts[i + 1])

    def _integrate_by_arc(self, x0, v0, u, dl_targets, direction, sub_dt):
        """Integrate matching arc lengths of center trajectory.
        direction: +1 forward, -1 backward in time.
        Returns (n+1, 3) pts starting from (x0,v0), and stuck flag.
        """
        model = self._ctx.model
        n = len(dl_targets)
        pts = np.zeros((n + 1, 3))
        pts[0] = [x0, Y_NODES, v0]
        max_sub = 200

        x, v = x0, v0
        stuck = False

        for j in range(n):
            if stuck:
                pts[j + 1] = [x, Y_NODES, v]
                continue

            target = dl_targets[j]
            if target < 1e-12:
                pts[j + 1] = [x, Y_NODES, v]
                continue

            accumulated = 0.0
            xc, vc = x, v

            for _ in range(max_sub):
                xn, vn = model.step(xc, vc, u, sub_dt * direction)
                step_dl = np.sqrt((xn - xc)**2 + (vn - vc)**2)
                if step_dl < 1e-12:
                    stuck = True
                    break
                if accumulated + step_dl >= target:
                    frac = (target - accumulated) / step_dl
                    xc = xc + frac * (xn - xc)
                    vc = vc + frac * (vn - vc)
                    break
                accumulated += step_dl
                xc, vc = xn, vn
            else:
                stuck = True

            x, v = xc, vc
            pts[j + 1] = [x, Y_NODES, v]

        return pts, stuck

    def _integrate_by_ortho(self, x0, v0, u, prev_xv, direction, dtau, n):
        """Integrate clone, optimizing front orthogonality to prev trajectory.
        prev_xv: (n+1, 2) — [(x,v), ...] of the neighbor closer to center.
        direction: +1 forward, -1 backward in time.
        Returns (n+1, 3) pts starting from (x0,v0).
        """
        model = self._ctx.model
        pts = np.zeros((n + 1, 3))
        pts[0] = [x0, Y_NODES, v0]

        x, v = x0, v0

        for j in range(n):
            px, pv = prev_xv[j + 1]
            fpx, fpv = model.deriv(px, pv, u)

            def cost(t):
                xc, vc = model.step(x, v, u, t * direction)
                dx, dv = xc - px, vc - pv
                dot_p = dx * fpx + dv * fpv
                fcx, fcv = model.deriv(xc, vc, u)
                dot_c = dx * fcx + dv * fcv
                return dot_p**2 + dot_c**2

            res = minimize_scalar(cost, bounds=(0.05 * dtau, 10.0 * dtau), method='bounded')
            x, v = model.step(x, v, u, res.x * direction)
            pts[j + 1] = [x, Y_NODES, v]

        return pts

    def _recompute(self) -> None:
        pm = self._ctx.param_manager
        n = self._n_tau
        ns = self._n_s
        a_max = pm.a_max
        dtau = pm.tau / n
        ds = pm.r_s / ns if ns > 0 else 0.0

        u = self._u_sign * a_max

        lp = self._ctx.look_point
        x0, v0 = float(lp[0]), float(lp[1])

        # build starting points: center at index ns, step ortho outward
        width = 2 * ns + 1
        starts = np.zeros((width, 2))
        starts[ns] = [x0, v0]

        cx, cv = x0, v0
        for i in range(ns):
            cx, cv = self._ortho_step(cx, cv, u, ds, +1)
            starts[ns + i + 1] = [cx, cv]

        cx, cv = x0, v0
        for i in range(ns):
            cx, cv = self._ortho_step(cx, cv, u, ds, -1)
            starts[ns - i - 1] = [cx, cv]

        # integrate center trajectory (always by time)
        all_pts = [None] * width
        center_pts = self._integrate_trajectory(
            starts[ns, 0], starts[ns, 1], u, dtau, n,
            self._traj_nodes[ns], self._traj_lines[ns])
        all_pts[ns] = center_pts

        mode = self._mode

        if mode == 0 or ns == 0:
            # TIME: all clones step by same dtau
            for oi in range(width):
                if oi == ns:
                    continue
                pts = self._integrate_trajectory(
                    starts[oi, 0], starts[oi, 1], u, dtau, n,
                    self._traj_nodes[oi], self._traj_lines[oi])
                all_pts[oi] = pts

        elif mode == 1:
            # ARC: clones match center arc lengths
            dl_fwd = np.zeros(n)
            dl_bwd = np.zeros(n)
            for j in range(n):
                dx = center_pts[n + j + 1, 0] - center_pts[n + j, 0]
                dv = center_pts[n + j + 1, 2] - center_pts[n + j, 2]
                dl_fwd[j] = np.sqrt(dx * dx + dv * dv)
            for j in range(n):
                dx = center_pts[n - j - 1, 0] - center_pts[n - j, 0]
                dv = center_pts[n - j - 1, 2] - center_pts[n - j, 2]
                dl_bwd[j] = np.sqrt(dx * dx + dv * dv)

            sub_dt = abs(dtau) / 20
            for oi in range(width):
                if oi == ns:
                    continue
                sx, sv = starts[oi]
                fwd_pts, _ = self._integrate_by_arc(sx, sv, u, dl_fwd, +1, sub_dt)
                bwd_pts, _ = self._integrate_by_arc(sx, sv, u, dl_bwd, -1, sub_dt)

                pts = np.zeros((2 * n + 1, 3))
                pts[n] = [sx, Y_NODES, sv]
                pts[n + 1:] = fwd_pts[1:]
                pts[:n] = bwd_pts[1:][::-1]

                self._update_visual(oi, pts)
                all_pts[oi] = pts

        elif mode == 2:
            # ORTHO: clones optimize front orthogonality, from center outward
            for dist in range(1, ns + 1):
                for oi in [ns + dist, ns - dist]:
                    if oi < 0 or oi >= width:
                        continue
                    prev_oi = oi - 1 if oi > ns else oi + 1
                    sx, sv = starts[oi]

                    prev_fwd_xv = all_pts[prev_oi][n:, [0, 2]]
                    fwd_pts = self._integrate_by_ortho(sx, sv, u, prev_fwd_xv, +1, dtau, n)

                    prev_bwd_xv = all_pts[prev_oi][n::-1, [0, 2]]
                    bwd_pts = self._integrate_by_ortho(sx, sv, u, prev_bwd_xv, -1, dtau, n)

                    pts = np.zeros((2 * n + 1, 3))
                    pts[n] = [sx, Y_NODES, sv]
                    pts[n + 1:] = fwd_pts[1:]
                    pts[:n] = bwd_pts[1:][::-1]

                    self._update_visual(oi, pts)
                    all_pts[oi] = pts

        # front lines at every time step
        for ti in range(2 * n + 1):
            for i, ln in enumerate(self._front_lines[ti]):
                self._lm.update(ln, all_pts[i][ti], all_pts[i + 1][ti])

        # surfaces
        Y_SURF = 0.01
        if self._surf_fwd:
            fwd_grid = np.zeros((width, n + 1, 3))
            for oi in range(width):
                fwd_grid[oi] = all_pts[oi][n:]
            fwd_grid[:, :, 1] = Y_SURF
            self._sfm.update_grid(self._surf_fwd, fwd_grid)

        if self._surf_bwd:
            bwd_grid = np.zeros((width, n + 1, 3))
            for oi in range(width):
                bwd_grid[oi] = all_pts[oi][:n + 1]
            bwd_grid[:, :, 1] = Y_SURF
            self._sfm.update_grid(self._surf_bwd, bwd_grid)

    def tick(self) -> None:
        if not self._active:
            return

        pm = self._ctx.param_manager
        n = int(pm.n_tau)
        ns = int(pm.n_s)

        if n != self._n_tau or ns != self._n_s:
            self._build()
            self._cached_root = np.full(2, np.nan)

        if self._n_tau == 0:
            return

        lp = self._ctx.look_point
        a_max = pm.a_max
        tau = pm.tau
        r_s = pm.r_s

        dirty = (
            not np.array_equal(lp, self._cached_root)
            or a_max != self._cached_a_max
            or tau != self._cached_tau
            or r_s != self._cached_r_s
        )
        if not dirty:
            return

        self._cached_root = np.array(lp, copy=True)
        self._cached_a_max = a_max
        self._cached_tau = tau
        self._cached_r_s = r_s

        self._recompute()
