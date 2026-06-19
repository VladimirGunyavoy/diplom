"""
ScalableTipArrow - A directional line with triangle arrowhead at the endpoint
==============================================================================
Like ScalableArrow but the triangle is at p2 (tip), not at the midpoint.
"""

import numpy as np
from ursina import Entity, Mesh, Vec3
from .scalable_line import ScalableLine


class ScalableTipArrow(ScalableLine):

    def __init__(self, p1, p2, thickness=2, **kwargs):
        self._tri_entity = None
        super().__init__(p1, p2, thickness=thickness, **kwargs)
        tri_mesh = Mesh(
            vertices=[Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)],
            triangles=[[0, 1, 2]],
            mode='triangle'
        )
        self._tri_entity = Entity(model=tri_mesh, double_sided=True, alpha=1)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ('color', 'enabled'):
            tri = object.__getattribute__(self, '_tri_entity') if '_tri_entity' in self.__dict__ else None
            if tri is not None:
                setattr(tri, name, value)
                if name == 'color':
                    tri.alpha = 1.0

    def apply_transform(self, a: float, b: np.ndarray, **kwargs) -> None:
        super().apply_transform(a, b, **kwargs)
        p1 = self.real_p1 * a + b
        p2 = self.real_p2 * a + b
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-8:
            return
        d_unit = d / length
        size = length / 4
        half_base = size / np.sqrt(15)
        perp = np.array([-d_unit[2], 0.0, d_unit[0]])

        tip = p2.copy()
        tip[1] += 0.02
        base_center = tip - d_unit * size
        base_l = base_center + perp * half_base
        base_r = base_center - perp * half_base

        self._tri_entity.model.vertices = [Vec3(*tip), Vec3(*base_l), Vec3(*base_r)]
        self._tri_entity.model.generate()
