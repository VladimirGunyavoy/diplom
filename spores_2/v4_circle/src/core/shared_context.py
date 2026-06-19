"""
SharedContext - Live data container
=====================================

Holds frequently updated variables shared between managers and game objects.
Writers bind getters (lambdas) by key; SharedContext pulls data from them each frame.
Readers access data as attributes: ctx.look_point
"""

from typing import Callable, Any


class SharedContext:

    def __init__(self):
        self._bindings: dict[str, Callable] = {}

    def bind(self, key: str, getter: Callable, default: Any = None) -> None:
        """Register a getter for a live value. Accessible as ctx.<key> after first update()."""
        self._bindings[key] = getter
        if not hasattr(self, key):
            setattr(self, key, default)

    def tick(self) -> None:
        for key, getter in self._bindings.items():
            setattr(self, key, getter())
