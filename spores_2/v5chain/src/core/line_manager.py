"""
LineManager - Factory and registry for ScalableLine objects
============================================================
"""

import numpy as np
from typing import Dict, TYPE_CHECKING
from .scalable_line import ScalableLine
from .scalable_arrow import ScalableArrow

if TYPE_CHECKING:
    from .zoom_manager import ZoomManager


class LineManager:

    def __init__(self, zoom_manager: "ZoomManager"):
        self._zoom_manager = zoom_manager
        self._lines: Dict[str, ScalableLine] = {}
        self.arrow_scale: float = 1.0

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

    def create_arrow(self, name: str, p1, p2, color, alpha: float = 1.0, t_sign: int = 1) -> ScalableArrow:
        arrow = ScalableArrow(p1=p1, p2=p2, t_sign=t_sign)
        arrow.color = color
        arrow.alpha = 1.0
        arrow.size_factor = self.arrow_scale
        self._zoom_manager.register_object(arrow, name=name)
        self._lines[name] = arrow
        return arrow

    def disable(self, name: str) -> None:
        if name in self._lines:
            self._lines[name].enabled = False
            self._zoom_manager.unregister_object(name)

    def set_visible(self, name: str, visible: bool) -> None:
        self._lines[name].enabled = visible

    def increase_arrow_size(self, factor: float = 1.2) -> None:
        self.arrow_scale *= factor
        self._apply_arrow_scale()
        print(f"[LineManager] Arrow scale: {self.arrow_scale:.3f}")

    def decrease_arrow_size(self, factor: float = 1.2) -> None:
        self.arrow_scale /= factor
        self._apply_arrow_scale()
        print(f"[LineManager] Arrow scale: {self.arrow_scale:.3f}")

    def _apply_arrow_scale(self) -> None:
        for line in self._lines.values():
            if isinstance(line, ScalableArrow):
                line.size_factor = self.arrow_scale
        self._zoom_manager.update_transform()
