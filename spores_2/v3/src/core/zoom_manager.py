from ursina import *
import numpy as np
from typing import Dict, Optional, Tuple

from .scalable import Scalable
from .color_manager import ColorManager

# Use TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .scene_manager import SceneManager

class ZoomManager:
    def __init__(self, scene_setup: 'SceneManager',
                 color_manager: Optional[ColorManager] = None):
        self.zoom_fact: float = 1 + 1/8

        self.a_transformation: float = 1.0
        self.b_translation: np.ndarray = np.array([0, 0, 0], dtype=float)

        # Use provided ColorManager or create new one
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()

        self.objects: Dict[str, Scalable] = {}
        self.scene_setup: 'SceneManager' = scene_setup

        self.invariant_point: Tuple[float, float, float] = (0, 0, 0)

    def register_object(self, obj: Scalable, name: Optional[str] = None) -> None:
        """Register object in zoom manager."""
        if name is None:
            name = f"obj_{len(self.objects)}"
        self.objects[name] = obj
        obj.apply_transform(self.a_transformation, self.b_translation)

    def unregister_object(self, name: str) -> None:
        """Remove object from zoom manager."""
        if name in self.objects:
            del self.objects[name]

    def identify_invariant_point(self) -> Tuple[float, float]:
        """Calculate look point where camera ray intersects ground (y=0)."""
        player = self.scene_setup.player
        psi = np.radians(self.scene_setup.player.rotation_y)
        phi = np.radians(self.scene_setup.player.camera_pivot.rotation_x)

        h = self.scene_setup.player.camera_pivot.world_position.y

        if abs(np.tan(phi)) < 0.001:
            # Avoid division by zero if camera is looking straight ahead
            return 0, 0

        d = (h / np.tan(phi))

        dx = d * np.sin(psi)
        dy = d * np.cos(psi)

        x_0 = self.scene_setup.player.camera_pivot.world_position.x + dx
        z_0 = self.scene_setup.player.camera_pivot.world_position.z + dy

        self.invariant_point = np.array([x_0, z_0])
        return x_0, z_0

    @property
    def real_look_point(self) -> np.ndarray:
        """Invariant point in real (pre-transform) coordinates."""
        ip = self.invariant_point
        return np.array([
            (ip[0] - self.b_translation[0]) / self.a_transformation,
            (ip[1] - self.b_translation[2]) / self.a_transformation,
        ])

    def tick(self) -> None:
        self.identify_invariant_point()

    def update_transform(self) -> None:
        """Apply current transformation to all registered objects."""
        for obj in self.objects.values():
            try:
                if hasattr(obj, 'position'):
                    obj.apply_transform(self.a_transformation, self.b_translation)
            except (AssertionError, AttributeError, RuntimeError) as e:
                # Object is invalid - skip without crash
                continue

    def change_zoom(self, sign: int) -> None:
        """Change zoom level while maintaining invariant point."""
        inv = np.array(self.identify_invariant_point())
        inv_3d = np.array([inv[0], 0, inv[1]])

        zoom_multiplier = self.zoom_fact ** sign
        self.a_transformation *= zoom_multiplier
        self.b_translation = zoom_multiplier * self.b_translation + (1 - zoom_multiplier) * inv_3d
        self.update_transform()

    def reset_all(self) -> None:
        """Reset all transformations to initial state."""
        self.a_transformation = 1
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.update_transform()
        self.scene_setup.player.speed = int(self.scene_setup.base_speed)
        self.scene_setup.player.position = self.scene_setup.base_position

    def zoom_in(self) -> None:
        """Increases zoom (zoom in)."""
        self.change_zoom(1)

    def zoom_out(self) -> None:
        """Decreases zoom (zoom out)."""
        self.change_zoom(-1)

    def reset_zoom(self) -> None:
        """Resets all transformations to initial state."""
        self.reset_all()
