"""
Screen Manager - On-screen text display
========================================

Manages Message objects and updates them every frame via UpdateManager.
"""

import os, textwrap
from ursina import Text, window
from typing import Any, Callable, List, Tuple

WRAP_COLS = int(os.environ.get('V8_WRAP_COLS', 60))      # лимит длины строки подсказки/статуса (слово пользователя 2026-10-12); дополнительно режется по ширине окна
CHAR_W = 0.0137                                           # ширина символа при scale .7 в единицах UI Ursina (замер: «Look: [ 0.00 0.00]» 19 симв. = 0.26)
LINE_H = 0.025                                            # высота строки при scale .7


def wrap_text(text: str, cols: int = WRAP_COLS, indent: str = '    ') -> str:
    """перенос по словам: строки (\\n) длиннее cols разбиваются, продолжение — с отступом indent; слишком длинное слово рвётся"""
    out = []
    for line in str(text).split('\n'):
        out += textwrap.wrap(line, max(cols, 10), subsequent_indent=indent, break_long_words=True, break_on_hyphens=False, drop_whitespace=True) or [''] if len(line) > cols else [line]
    return '\n'.join(out)


REF_HALF = 16 / 9 / 2                                      # позиции сообщений заданы для окна 16:9 (±0.889 по x); для другой пропорции — от своего края


def edge_x(x0: float) -> float:
    """x для текущей пропорции окна: сообщение слева (x0 < 0) держит расстояние до левого края, справа — до правого"""
    asp = window.aspect_ratio
    x = -asp / 2 + (x0 + REF_HALF) if x0 < 0 else x0 + asp / 2 - REF_HALF
    return max(x, -asp / 2 + 0.01)                          # очень узкое окно: не левее края (строки сократятся по fit_cols)


def fit_cols(x: float, scale: float = 0.7, cols: int = WRAP_COLS) -> int:
    """число колонок: не больше cols и не дальше правого края окна (любой размер/пропорции: ширина UI = aspect_ratio, от −aspect/2 до +aspect/2)"""
    room = window.aspect_ratio / 2 - x - 0.02
    return max(20, min(cols, int(room / (CHAR_W * scale / 0.7))))


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
        wrap: bool = True,
        bottom: bool = False,
    ):
        self.name = name
        self.getter = getter
        self.wrap = wrap; self.bottom = bottom; self._last = None

        x = position[0] + offset[0]
        y = position[1] + offset[1]

        self._x, self._y = x, y
        self._text_entity = Text(
            text=self._text(),
            font=Text.default_font,
            position=(edge_x(x), y),
            origin=(-0.5, -0.5) if bottom else (-0.5, 0.5),                  # bottom: строки растут вверх от y (нижняя строка статуса не уходит за экран)
            scale=0.7,
        )

    def _text(self) -> str:
        t = self.getter()
        return wrap_text(t, fit_cols(edge_x(self._x))) if self.wrap else t

    def tick(self) -> None:
        self._text_entity.x = edge_x(self._x); t = self._text()
        if t != self._last: self._last = t; self._text_entity.text = t


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
