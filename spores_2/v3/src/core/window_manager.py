import json
import os
from ursina import window, color
from typing import Optional, Tuple

class WindowManager:
    """
    Simple class for managing Ursina window settings.
    Monitor configs are loaded from config/monitors.json.
    """

    _CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'monitors.json')

    def __init__(self, title: str = "Ursina App", monitor: str = "main", fullscreen: bool = False):
        """
        Initializes window manager.

        Args:
            title (str): Window title.
            monitor (str): Monitor type ("main", "top", "left", "down").
            fullscreen (bool): Whether to launch in fullscreen mode.
        """
        self.current_monitor = monitor
        self.fullscreen = fullscreen
        window.title = title

        with open(self._CONFIG_PATH, 'r') as f:
            self._monitors = json.load(f)

        if fullscreen:
            window.fullscreen = True
        else:
            self._apply_monitor(monitor)


    def _apply_monitor(self, monitor: str) -> None:
        config = self._monitors.get(monitor, self._monitors["main"])
        window.size = tuple(config["size"])
        window.position = tuple(config["position"])

    def get_current_monitor(self) -> str:
        """Returns current monitor name."""
        return self.current_monitor

    def get_margin(self) -> Tuple[float, float]:
        """Returns UI margin (x, y) for the current monitor from config."""
        config = self._monitors.get(self.current_monitor, self._monitors["main"])
        m = config.get("margin", [0.0, 0.0])
        return (m[0], m[1])

    def set_size(self, size: tuple) -> None:
        """Sets window size."""
        window.size = size

    def set_position(self, position: tuple) -> None:
        """Sets window position."""
        window.position = position

    def set_background_color(self, a_color: color) -> None:
        """Sets background color."""
        window.color = a_color

    def toggle_fullscreen(self) -> None:
        """Toggles fullscreen mode."""
        self.fullscreen = not self.fullscreen
        window.fullscreen = self.fullscreen

        if not self.fullscreen:
            self._apply_monitor(self.current_monitor)

    def set_fullscreen(self, fullscreen: bool) -> None:
        """Sets fullscreen mode."""
        self.fullscreen = fullscreen
        window.fullscreen = fullscreen

        if not fullscreen:
            self._apply_monitor(self.current_monitor)

    def is_fullscreen(self) -> bool:
        """Returns True if window is in fullscreen mode."""
        return self.fullscreen
