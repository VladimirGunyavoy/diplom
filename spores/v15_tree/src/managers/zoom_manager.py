from ursina import *
import numpy as np
from typing import Dict, Optional, Tuple

from ..core.spore import Spore
from ..utils.scalable import Scalable
from .color_manager import ColorManager
from ..visual.ui_manager import UIManager
from ..visual.scene_setup import SceneSetup
# from .link import Link

class ZoomManager:
    def __init__(self, scene_setup: SceneSetup, 
                 color_manager: Optional[ColorManager] = None, 
                 ui_manager: Optional[UIManager] = None):
        self.zoom_x: float = 1
        self.zoom_y: float = 1
        self.zoom_z: float = 1

        self.zoom_fact: float = 1 + 1/8

        self.a_transformation: float = 1.0
        self.b_translation: np.ndarray = np.array([0, 0, 0], dtype=float)

        self.spores_scale: float = 1.0
        self.common_scale: float = 1.0

        # Используем переданный ColorManager или создаем новый
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()

        # Используем переданный UIManager или создаем новый
        self.ui_manager: UIManager = ui_manager if ui_manager is not None else UIManager(self.color_manager)

        self.objects: Dict[str, Scalable] = {}
        self.scene_setup: SceneSetup = scene_setup

        # UI элементы для zoom manager теперь создаются в UI_setup.py
        self.ui_elements: Dict = {}

        self.invariant_point: Tuple[float, float, float] = (0, 0, 0)

    def register_object(self, obj: Scalable, name: Optional[str] = None) -> None:
        if name is None:
            name = f"obj_{len(self.objects)}"
        self.objects[name] = obj
        obj.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.spores_scale)
    
    def unregister_object(self, name: str) -> None:
        """Удаляет объект из менеджера масштабирования."""
        if name in self.objects:
            del self.objects[name]


    def identify_invariant_point(self) -> Tuple[float, float]:
        player = self.scene_setup.player
        psi = np.radians(self.scene_setup.player.rotation_y)
        phi = np.radians(self.scene_setup.player.camera_pivot.rotation_x)

        h = self.scene_setup.player.camera_pivot.world_position.y
        
        if np.tan(phi) == 0:
            # Avoid division by zero if camera is looking straight ahead
            return 0, 0

        d = (h / np.tan(phi)) 

        dx = d * np.sin(psi)
        dy = d * np.cos(psi) 

        x_0 = self.scene_setup.player.camera_pivot.world_position.x + dx
        z_0 = self.scene_setup.player.camera_pivot.world_position.z + dy

        # Обновление UI происходит автоматически через ui_manager.update_dynamic_elements()
        return x_0, z_0

    def update_transform(self) -> None:
        from src.visual.link import Link
        for obj in self.objects.values():
            try:
                # Проверяем что объект существует и имеет валидный NodePath
                if isinstance(obj, Link):
                    obj.update_geometry()
                if hasattr(obj, 'enabled') and obj.enabled and hasattr(obj, 'position'):
                    obj.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.spores_scale)
                
            except (AssertionError, AttributeError, RuntimeError) as e:
                # Объект невалиден - пропускаем без краша
                continue
        # self.scene_setup.player.speed = self.scene_setup.base_speed * self.a_transformation
    
    def change_zoom(self, sign: int) -> None:
        inv = np.array(self.identify_invariant_point())
        inv_3d = np.array([inv[0], 0, inv[1]])

        zoom_multiplier = self.zoom_fact ** sign
        self.a_transformation *= zoom_multiplier
        self.b_translation = zoom_multiplier * self.b_translation + (1 - zoom_multiplier) * inv_3d
        self.update_transform()

    def reset_all(self) -> None:
        self.a_transformation = 1
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.spores_scale = 1
        self.update_transform()
        self.scene_setup.player.speed = self.scene_setup.base_speed
        self.scene_setup.player.position = self.scene_setup.base_position



    def scale_spores(self, sign: int) -> None:
        self.spores_scale *= self.zoom_fact ** sign
        
        self.update_transform()

    def zoom_in(self) -> None:
        """Увеличивает масштаб (приближает)."""
        self.change_zoom(1)

    def zoom_out(self) -> None:
        """Уменьшает масштаб (отдаляет)."""
        self.change_zoom(-1)

    def reset_zoom(self) -> None:
        """Сбрасывает все трансформации к исходному состоянию."""
        self.reset_all()

    def increase_spores_scale(self) -> None:
        """Увеличивает масштаб только для спор."""
        self.scale_spores(1)

    def decrease_spores_scale(self) -> None:
        """Уменьшает масштаб только для спор."""
        self.scale_spores(-1)



