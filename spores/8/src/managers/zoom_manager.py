from ursina import *
import numpy as np
from src.core.spore import Spore
from src.utils.scalable import Scalable
from .color_manager import ColorManager
from src.ui.ui_manager import UIManager
# from .link import Link

class ZoomManager:
    def __init__(self, scene_setup, color_manager=None, ui_manager=None):
        self.zoom_x = 1
        self.zoom_y = 1
        self.zoom_z = 1

        self.zoom_fact = 1 + 1/8

        self.a_transformation = 1
        self.b_translation = np.array([0, 0, 0], dtype=float)

        self.spores_scale = 1
        self.common_scale = 1

        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager

        # Используем переданный UIManager или создаем новый
        if ui_manager is None:
            ui_manager = UIManager(color_manager)
        self.ui_manager = ui_manager

        self.objects = {}
        self.scene_setup = scene_setup

        # UI элементы для zoom manager теперь создаются в UI_setup.py
        self.ui_elements = {}


    def register_object(self, object: Scalable, name=None):
        if name is None:
            name = f"obj_{len(self.objects)}"
        self.objects[name] = object
        object.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.spores_scale)


    def identify_invariant_point(self):
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

    def update_transform(self):
        from src.scene.link import Link
        for obj in self.objects.values():
            if isinstance(obj, Link):
                obj.update_geometry()
            obj.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.spores_scale)
        # self.scene_setup.player.speed = self.scene_setup.base_speed * self.a_transformation
    
    def change_zoom(self, sign):
        inv = np.array(self.identify_invariant_point())
        inv_3d = np.array([inv[0], 0, inv[1]])

        zoom_multiplier = self.zoom_fact ** sign
        self.a_transformation *= zoom_multiplier
        self.b_translation = zoom_multiplier * self.b_translation + (1 - zoom_multiplier) * inv_3d
        self.update_transform()

    def reset_all(self):
        self.a_transformation = 1
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.spores_scale = 1
        self.update_transform()
        self.scene_setup.player.speed = self.scene_setup.base_speed
        self.scene_setup.player.position = self.scene_setup.base_position



    def scale_spores(self, sign):
        self.spores_scale *= self.zoom_fact ** sign
        
        self.update_transform()

    def input_handler(self, key):
        if key == 'e':
            self.change_zoom(1)
        elif key == 't':
            self.change_zoom(-1)
        elif key == 'r':
            self.reset_all()
        elif key == '1':
            self.scale_spores(1)
        elif key == '2':
            self.scale_spores(-1)

        



