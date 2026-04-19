"""
MyObject - Object moving in a circle
=====================================

A sphere that moves along a circular path.
Can change speed via input controls.
"""

import math
from ursina import color
from .scalable import Scalable


class MyObject(Scalable):
    """
    A sphere that moves along a circular path around origin.

    Attributes:
        radius: Radius of the circular path (default: 1.5)
        speed: Angular speed in radians per second (default: 1.0)
        angle: Current angle in radians
    """

    def __init__(
        self,
        radius: float = 1.5,
        speed: float = 1.0,
        color_value=color.yellow,
        **kwargs
    ):
        """
        Initialize MyObject.

        Args:
            radius: Radius of circular path
            speed: Angular speed (radians per second)
            color_value: Color of the sphere
        """
        # Initialize as Scalable sphere
        super().__init__(
            model='sphere',
            scale=1/5,
            color=color_value,
            **kwargs
        )

        self.radius = radius
        self.speed = speed
        self.angle = 0.0  # Start at angle 0

        # Set initial position
        self._update_position()

    def _update_position(self) -> None:
        """Update position based on current angle."""
        x = self.radius * math.cos(self.angle)
        z = self.radius * math.sin(self.angle)
        self.position = (x, 0, z)

    def update_position(self, dt: float) -> None:
        """
        Update object position (move along circle).

        Args:
            dt: Delta time in seconds
        """
        # Update angle based on speed
        self.angle += self.speed * dt

        # Keep angle in [0, 2π] range
        if self.angle >= 2 * math.pi:
            self.angle -= 2 * math.pi

        # Update position
        self._update_position()

    def increase_speed(self, delta: float = 0.5) -> None:
        """Increase angular speed."""
        self.speed += delta
        print(f"[MyObject] Speed increased to {self.speed:.2f} rad/s")

    def decrease_speed(self, delta: float = 0.5) -> None:
        """Decrease angular speed."""
        self.speed = max(0.0, self.speed - delta)  # Don't go negative
        print(f"[MyObject] Speed decreased to {self.speed:.2f} rad/s")
