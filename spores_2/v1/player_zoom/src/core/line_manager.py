"""
LineManager - Factory and registry for ScalableLine objects
============================================================
"""

import numpy as np
from typing import Dict, TYPE_CHECKING
from .scalable_line import ScalableLine

if TYPE_CHECKING:
    from .zoom_manager import ZoomManager


class LineManager:

    def __init__(self, zoom_manager: "ZoomManager"):
        self._zoom_manager = zoom_manager
        self._lines: Dict[str, ScalableLine] = {}

    def create(self, name: str, p1, p2, color, alpha: float = 1.0) -> ScalableLine:
        line = ScalableLine(p1=p1, p2=p2)
        line.color = color
        line.alpha = alpha
        self._zoom_manager.register_object(line, name=name)
        self._lines[name] = line
        return line

    def update(self, name: str, p1, p2) -> None:
        line = self._lines[name]
        line.real_p1 = np.array(p1, dtype=float)
        line.real_p2 = np.array(p2, dtype=float)

    def disable(self, name: str) -> None:
        if name in self._lines:
            self._lines[name].enabled = False
            self._zoom_manager.unregister_object(name)
