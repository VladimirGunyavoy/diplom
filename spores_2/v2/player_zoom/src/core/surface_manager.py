"""
SurfaceManager - Factory and registry for ScalableFanSurface objects
====================================================================
"""

import numpy as np
from typing import Dict, TYPE_CHECKING
from .scalable_surface import ScalableFanSurface

if TYPE_CHECKING:
    from .zoom_manager import ZoomManager


class SurfaceManager:

    def __init__(self, zoom_manager: "ZoomManager", dh: float = 0.0):
        self._zoom_manager = zoom_manager
        self._surfaces: Dict[str, ScalableFanSurface] = {}
        self._dh = dh
        self._count = 0

    def create_fan(self, name: str, n_tau: int, color, alpha: float = 0.2) -> ScalableFanSurface:
        y_offset = self._count * self._dh
        self._count += 1
        surf = ScalableFanSurface(n_tau=n_tau, y_offset=y_offset)
        surf.color = color
        surf.alpha = alpha
        surf.double_sided = True
        self._zoom_manager.register_object(surf, name=name)
        self._surfaces[name] = surf
        return surf

    def update_fan(self, name: str, center: np.ndarray, positions_grid: np.ndarray) -> None:
        self._surfaces[name].update_points(center, positions_grid)

    def disable(self, name: str) -> None:
        if name in self._surfaces:
            self._surfaces[name].enabled = False
            self._zoom_manager.unregister_object(name)
