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
