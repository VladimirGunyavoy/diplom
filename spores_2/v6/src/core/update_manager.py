"""
Update Manager - Centralized update handler
===========================================

Manages all per-frame updates in a centralized way.
Based on v16_picker UpdateManager but simplified for player_zoom sandbox.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .scene_manager import SceneManager
    from .zoom_manager import ZoomManager
    from .input_manager import InputManager
    from .object_manager import ObjectManager
    from .screen_manager import ScreenManager
    from .shared_context import SharedContext


class UpdateManager:
    """
    Centralized class for calling update methods every frame.
    Simplified version adapted for player_zoom sandbox.

    Components can be registered after initialization to avoid circular dependencies.
    """

    def __init__(self):
        self.scene_setup: Optional["SceneManager"] = None
        self.zoom_manager: Optional["ZoomManager"] = None
        self.input_manager: Optional["InputManager"] = None
        self.object_manager: Optional["ObjectManager"] = None
        self.screen_manager: Optional["ScreenManager"] = None
        self.shared_context: Optional["SharedContext"] = None

    def register(self, **kwargs) -> None:
        for name, component in kwargs.items():
            if not hasattr(self, name):
                raise ValueError(f"[UpdateManager] Unknown component: '{name}'")
            setattr(self, name, component)

    def update_all(self) -> None:
        for component in [
            self.input_manager,   # process input state
            self.scene_setup,     # player moves first
            self.zoom_manager,    # look_point depends on player position
            self.shared_context,  # pull fresh data before object ticks
            self.object_manager,  # tick objects + re-apply zoom transforms
            self.screen_manager,
        ]:
            if component:
                component.tick()
