"""
ScalableSurface - A triangulated mesh surface that reacts to zoom transforms
=============================================================================

Builds a surface from a grid of points: N rows x M columns.
Designed for trajectory visualization.
"""

import numpy as np
from ursina import Mesh, Vec3
from .scalable import Scalable


def _build_triangles(N, M):
    """Build triangle indices for an N x M grid of vertices."""
    tris = []
    for i in range(N - 1):
        for j in range(M - 1):
            a = i * M + j
            b = (i + 1) * M + j
            c = i * M + (j + 1)
            d = (i + 1) * M + (j + 1)
            tris.append((a, b, c))
            tris.append((b, d, c))
    return tris


def _build_edges(N, M):
    """Build edge indices (wireframe) for an N x M grid of vertices.

    Two types of edges:
    - trajectory lines: consecutive states within each trajectory
    - inter-trajectory lines: connecting state j of trajectory i to state j of trajectory i+1
      (only immediate neighbors; first/last trajectory has 1 neighbor, others have 2)
    """
    edges = []
    # рёбра вдоль каждой траектории
    for i in range(N):
        for j in range(M - 1):
            edges.append((i * M + j, i * M + j + 1))
    return edges


class ScalableSurface(Scalable):

    def __init__(self, grid_points, wireframe=False, **kwargs):
        """
        Args:
            grid_points: list of N lists, each with M points [x, y, z]
                         (e.g. N trajectories x M states per trajectory)
        """
        self._N = len(grid_points)
        self._M = len(grid_points[0])

        flat = [pt for row in grid_points for pt in row]
        self.vertices_real = np.array(flat, dtype=float)  # shape (N*M, 3)

        if wireframe:
            indices = _build_edges(self._N, self._M)
            mode = 'line'
        else:
            indices = _build_triangles(self._N, self._M)
            mode = 'triangle'
        mesh = Mesh(
            vertices=[Vec3(*v) for v in self.vertices_real],
            triangles=indices,
            mode=mode,
        )
        super().__init__(model=mesh, **kwargs)

    def apply_transform(self, a: float, b: np.ndarray, **kwargs) -> None:
        verts = self.vertices_real * a + b
        self.model.vertices = [Vec3(*v) for v in verts]
        self.model.generate()


class ScalableFanSurface(Scalable):
    """Grid-triangulated fan surface: root + positions[k, step] grid.

    Triangulation:
      - fan:  root → positions[k,0]..positions[k+1,0]  (n_tau triangles)
      - grid: quads between adjacent rays k and k+1     (2*n_tau*(n_tau-1) triangles)

    y_offset: small vertical shift to avoid z-fighting between overlapping surfaces.
    """

    def __init__(self, n_tau: int, y_offset: float = 0.0, **kwargs):
        self._n_tau = n_tau
        self._y_offset = y_offset
        # v[0] = root,  v[1 + k*n_tau + j] = positions[k, j]
        n_verts = 1 + (n_tau + 1) * n_tau
        self.vertices_real = np.zeros((n_verts, 3), dtype=float)

        tris = [(0, 1 + k * n_tau, 1 + (k + 1) * n_tau) for k in range(n_tau)]
        for k in range(n_tau):
            for j in range(n_tau - 1):
                a = 1 + k * n_tau + j
                b = 1 + (k + 1) * n_tau + j
                c = 1 + k * n_tau + (j + 1)
                d = 1 + (k + 1) * n_tau + (j + 1)
                tris.append((a, b, c))
                tris.append((b, d, c))
        self._tris = tris

        mesh = Mesh(
            vertices=[Vec3(0, 0, 0)] * n_verts,
            triangles=self._tris,
            mode='triangle',
        )
        super().__init__(model=mesh, **kwargs)

    def update_points(self, center: np.ndarray, positions_grid: np.ndarray) -> None:
        """center: (3,), positions_grid: (n_tau+1, n_tau, 3)."""
        self.vertices_real[0] = center
        self.vertices_real[1:] = positions_grid.reshape(-1, 3)
        self.vertices_real[:, 1] += self._y_offset

    def apply_transform(self, a: float, b: np.ndarray, **kwargs) -> None:
        verts = self.vertices_real * a + b
        self.model.triangles = self._tris
        self.model.vertices = [Vec3(*v) for v in verts]
        self.model.generate()
