from ursina import window, color
from typing import Optional

class WindowManager:
    """
    Simple class for managing Ursina window settings.
    """

    # Settings for different monitors
    MONITORS = {
        "main": {"size": (1920, 1080), "position": (0, 0)},
        "top": {"size": (1920, 1080), "position": (0, -1080)},
        "left": {"size": (1800, 950), "position": (-1850, 250)},
        "down": {"size": (3000, 1700), "position": (-500, 1500)}
    }

    def __init__(self, title: str = "Ursina App", monitor: str = "main", fullscreen: bool = False):
        """
        Initializes window manager.

        Args:
            title (str): Window title.
            monitor (str): Monitor type ("main", "top", "left", "down").
            fullscreen (bool): Whether to launch in fullscreen mode.
        """
        self.current_monitor = monitor  # Save current monitor
        self.fullscreen = fullscreen
        window.title = title

        # Apply monitor settings
        if fullscreen:
            # In fullscreen mode use screen size
            window.fullscreen = True
        else:
            config = self.MONITORS.get(monitor, self.MONITORS["main"])
            window.size = config["size"]
            window.position = config["position"]


    def get_current_monitor(self) -> str:
        """Returns current monitor name."""
        return self.current_monitor

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
            # Return to monitor settings when exiting fullscreen mode
            config = self.MONITORS.get(self.current_monitor, self.MONITORS["main"])
            window.size = config["size"]
            window.position = config["position"]

    def set_fullscreen(self, fullscreen: bool) -> None:
        """Sets fullscreen mode."""
        self.fullscreen = fullscreen
        window.fullscreen = fullscreen

        if not fullscreen:
            # Return to monitor settings when exiting fullscreen mode
            config = self.MONITORS.get(self.current_monitor, self.MONITORS["main"])
            window.size = config["size"]
            window.position = config["position"]

    def is_fullscreen(self) -> bool:
        """Returns True if window is in fullscreen mode."""
        return self.fullscreen
