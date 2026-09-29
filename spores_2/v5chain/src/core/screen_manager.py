"""
Screen Manager - On-screen text display
========================================

Manages Message objects and updates them every frame via UpdateManager.
"""

from ursina import Text
from typing import Any, Callable, List, Tuple


class Message:
    """
    A single on-screen text message with a dynamic getter.

    Args:
        name:     Identifier for this message.
        position: Base screen position (x, y) in Ursina UI coordinates.
        getter:   Callable returning the string to display each frame.
        offset:   Per-message correction for monitor/setup differences.
    """

    def __init__(
        self,
        name: str,
        position: Tuple[float, float],
        getter: Callable[[], str],
        offset: Tuple[float, float] = (0.0, 0.0),
    ):
        self.name = name
        self.getter = getter

        x = position[0] + offset[0]
        y = position[1] + offset[1]

        self._text_entity = Text(
            text=getter(),
            font='VeraMono.ttf',
            position=(x, y),
            origin=(-0.5, 0.5),
            scale=0.7,
        )

    def tick(self) -> None:
        self._text_entity.text = self.getter()


class ScreenManager:
    """
    Holds all Message objects and updates them every frame.

    Register with UpdateManager so update() is called automatically.
    """

    def __init__(self):
        self._messages: List[Message] = []

    def add_message(self, message: Message) -> None:
        self._messages.append(message)
        print(f"[ScreenManager] Added message: {message.name}")

    def add_bindings_help(self, source: Any, position: Tuple[float, float], offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Add a message that displays all bindings from any object with get_help()."""
        self.add_message(Message(
            name='bindings_help',
            position=position,
            offset=offset,
            getter=source.get_help,
        ))

    def tick(self) -> None:
        for message in self._messages:
            message.tick()
