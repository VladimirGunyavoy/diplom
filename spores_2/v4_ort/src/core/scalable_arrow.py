"""
ScalableArrow - A directional line with triangle arrowhead at midpoint
======================================================================
Inherits from ScalableLine. Draws a small filled triangle at the midpoint
of the segment. Direction is controlled by t_sign:
  t_sign > 0 : arrow points p1 → p2
  t_sign < 0 : arrow points p2 → p1 (time goes backward)
Triangle size = 1/4 of segment length.
"""

import numpy as np
from ursina import Entity, Mesh, Vec3
from .scalable_line import ScalableLine


class ScalableArrow(ScalableLine):

    def __init__(self, p1, p2, t_sign: int = 1, thickness=3, **kwargs):
        self._tri_entity = None
        self.t_sign = t_sign
        self.size_factor: float = 1.0
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
        if self.t_sign < 0:
            d = -d
        d = d / length
        size = length / 4 * self.size_factor
        half_base = size / np.sqrt(15)  # leg = 2*base → half_base = size/sqrt(15)
        perp = np.array([-d[2], 0.0, d[0]])
        mid = (p1 + p2) / 2
        mid[1] += 0.0
        tip    = mid + d * (size * 2 / 3)
        base_l = mid - d * (size / 3) + perp * half_base
        base_r = mid - d * (size / 3) - perp * half_base
        self._tri_entity.model.vertices = [Vec3(*tip), Vec3(*base_l), Vec3(*base_r)]
        self._tri_entity.model.generate()
