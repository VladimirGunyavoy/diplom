"""
InputManager for player_zoom
Handles system commands, zoom, and user-defined bindings (press and hold+scroll).
"""

from ursina import held_keys, application
from typing import Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .scene_manager import SceneManager
    from .zoom_manager import ZoomManager
    from .window_manager import WindowManager
    from .object_manager import ObjectManager


@dataclass
class _Binding:
    key: str
    action: Callable
    mode: str                          # 'press' or 'scroll'
    description: str
    value_getter: Optional[Callable] = None


class InputManager:

    def __init__(self):
        self.scene_setup: Optional["SceneManager"] = None
        self.zoom_manager: Optional["ZoomManager"] = None
        self.window_manager: Optional["WindowManager"] = None
        self.object_manager: Optional["ObjectManager"] = None

        self._bindings: list[_Binding] = []

        print(f"[DEBUG] InputManager initialized")

    def register(self, **kwargs) -> None:
        for name, component in kwargs.items():
            if not hasattr(self, name):
                raise ValueError(f"[InputManager] Unknown component: '{name}'")
            setattr(self, name, component)
            print(f"   {name}: registered")

    def bind(self, key: str, action: Callable, mode: str = 'press', description: str = '',
             value_getter: Optional[Callable] = None) -> None:
        """
        Register a user binding.
        mode='press'  — action() called on key press
        mode='scroll' — action(sign) called on scroll while key is held; suppresses zoom
        value_getter  — optional callable returning current value (shown in scroll help)
        """
        self._bindings.append(_Binding(key=key, action=action, mode=mode,
                                       description=description, value_getter=value_getter))

    def get_help(self) -> str:
        lines = []
        scroll_bindings = [b for b in self._bindings if b.mode == 'scroll' and b.description]
        press_bindings  = [b for b in self._bindings if b.mode == 'press'  and b.description]

        if scroll_bindings:
            lines.append('scroll +')
            for b in scroll_bindings:
                value = f' - {b.value_getter():.3f}' if b.value_getter else ''
                lines.append(f'  {b.key} - {b.description}{value}')

        for b in press_bindings:
            lines.append(f'{b.key} - {b.description}')

        return '\n'.join(lines)

    def handle_input(self, key: str) -> None:

        # === SYSTEM ===
        if key == 'escape':
            application.quit()
            return

        if key == 'f11' and self.window_manager:
            self.window_manager.toggle_fullscreen()
            print(f"   [Window] Fullscreen: {'enabled' if self.window_manager.is_fullscreen() else 'disabled'}")
            return

        if key == 'alt' and self.scene_setup:
            self.scene_setup.toggle_freeze()
            return

        if self.scene_setup and self.scene_setup.input_frozen:
            return

        # === SCROLL: params take priority over zoom ===
        if key in ('scroll up', 'scroll down'):
            sign = +1 if key == 'scroll up' else -1
            for b in self._bindings:
                if b.mode == 'scroll' and held_keys[b.key]:
                    b.action(sign)
                    return
            if self.zoom_manager:
                if sign == +1:
                    self.zoom_manager.zoom_in()
                else:
                    self.zoom_manager.zoom_out()
            return

        # === ZOOM ===
        if self.zoom_manager:
            if key == 'e':
                self.zoom_manager.zoom_in()
                print("   [Zoom] Zoom in")
                return
            if key == 'q':
                self.zoom_manager.zoom_out()
                print("   [Zoom] Zoom out")
                return
            if key == 'r':
                self.zoom_manager.reset_zoom()
                print("   [Zoom] Reset")
                return

        # === USER BINDINGS (press) ===
        for b in self._bindings:
            if b.mode == 'press' and key == b.key:
                b.action()
                return

        # === FRAME ===
        if key == 'u' and self.scene_setup:
            self.scene_setup.toggle_frame()
            return

        # === DEBUG ===
        if key == 'h':
            self._print_debug_info()
            return

    def _print_debug_info(self):
        print("\n" + "=" * 50)
        print("DEBUG INFO")
        print("=" * 50)

        if self.scene_setup:
            print(f"Camera position: {self.scene_setup.player.position}")
            print(f"Camera rotation: y={self.scene_setup.player.rotation_y}, "
                  f"x={self.scene_setup.player.camera_pivot.rotation_x}")
            print(f"Cursor locked: {self.scene_setup.cursor_locked}")
            print(f"Input frozen: {self.scene_setup.input_frozen}")

        if self.zoom_manager:
            print(f"Zoom transform: a={self.zoom_manager.a_transformation:.4f}")
            print(f"Zoom translation: {self.zoom_manager.b_translation}")
            print(f"Registered objects: {len(self.zoom_manager.objects)}")
            look_x, look_z = self.zoom_manager.identify_invariant_point()
            print(f"Look point: ({look_x:.4f}, {look_z:.4f})")

        if self.scene_setup:
            print(f"Frame visible: {self.scene_setup.frame.is_visible()}")

        print("=" * 50 + "\n")

    def tick(self):
        pass
