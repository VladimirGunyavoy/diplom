"""
SporeManager - Manages visual size of all Spore objects
=========================================================

Stores current size and updates all registered spores at once.
Acts as a creation proxy to ObjectManager for Spore subclasses.
"""

import numpy as np
from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .spore import Spore
    from ..core.zoom_manager import ZoomManager
    from ..core.object_manager import ObjectManager


class SporeManager:

    def __init__(self, zoom_manager: "ZoomManager", object_manager: "ObjectManager"):
        self._spores: Dict[str, "Spore"] = {}
        self._zoom_manager = zoom_manager
        self._object_manager = object_manager
        self.size: float = 1.0

    def create(self, cls: Type["Spore"], name: str, **kwargs) -> "Spore":
        """Create a Spore via ObjectManager and register it here."""
        obj = self._object_manager.create(cls=cls, name=name, **kwargs)
        self.register(name, obj)
        return obj

    def register(self, name: str, spore: "Spore") -> None:
        if not self._spores:
            self.size = spore.scale.x
        else:
            spore.real_scale = np.array([self.size, self.size, self.size])
        self._spores[name] = spore

    def get(self, name: str) -> "Spore":
        return self._spores[name]

    def increase_size(self, factor: float = 1.2) -> None:
        self.size *= factor
        self._update_all()
        print(f"[SporeManager] Size: {self.size:.3f}")

    def decrease_size(self, factor: float = 1.2) -> None:
        self.size /= factor
        self._update_all()
        print(f"[SporeManager] Size: {self.size:.3f}")

    def _update_all(self) -> None:
        for spore in self._spores.values():
            spore.real_scale = np.array([self.size, self.size, self.size])
        self._zoom_manager.update_transform()
