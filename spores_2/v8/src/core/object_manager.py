"""
ObjectManager - Factory and registry for game objects
======================================================

Creates game objects and registers them in all necessary managers at once.
"""

from typing import List, Type, TYPE_CHECKING
from .scalable import GameObject
from ..spores.spore import Spore

if TYPE_CHECKING:
    from .zoom_manager import ZoomManager
    from .shared_context import SharedContext


class ObjectManager:

    def __init__(self, zoom_manager: "ZoomManager", shared_context: "SharedContext"):
        self.zoom_manager = zoom_manager
        self._shared_context = shared_context
        self._objects: List[GameObject] = []
        self._tickables: list = []

    def create(self, cls: Type[GameObject], name: str, **kwargs) -> GameObject:
        """Create game object and register it in zoom_manager. Auto-injects ctx for Spore subclasses."""
        if issubclass(cls, Spore) and 'ctx' not in kwargs:
            kwargs['ctx'] = self._shared_context
        obj = cls(**kwargs)
        self.zoom_manager.register_object(obj, name=name)
        self._objects.append(obj)
        return obj

    def register_tickable(self, obj) -> None:
        """Register any object with tick() to be called after game objects each frame."""
        self._tickables.append(obj)

    def tick(self) -> None:
        for obj in self._objects:
            obj.tick()
        for obj in self._tickables:
            obj.tick()
        self.zoom_manager.update_transform()
