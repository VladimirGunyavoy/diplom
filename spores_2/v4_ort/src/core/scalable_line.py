"""
ScalableLine - A line segment that reacts to zoom transforms
=============================================================
"""

import numpy as np
from ursina import Mesh, Vec3
from .scalable import Scalable


class ScalableLine(Scalable):

    def __init__(self, p1, p2, thickness=3, **kwargs):
        self.real_p1 = np.array(p1, dtype=float)
        self.real_p2 = np.array(p2, dtype=float)
        mesh = Mesh(vertices=[Vec3(*p1), Vec3(*p2)], mode='line', thickness=thickness)
        super().__init__(model=mesh, **kwargs)

    def apply_transform(self, a: float, b: np.ndarray, **kwargs) -> None:
        p1 = self.real_p1 * a + b
        p2 = self.real_p2 * a + b
        self.model.vertices = [Vec3(*p1), Vec3(*p2)]
        self.model.generate()
