from ursina import *
import numpy as np
from .spore import Spore
from .scalable import Scalable
from .color_manager import ColorManager
# from .link import Link

class ZoomManager:
    def __init__(self, scene_setup, color_manager=None):
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

        self.txt = Text(text='', position=(0, 0, 0))

        self.objects = {}

        self.scene_setup = scene_setup
        self.instructions = Text(
            text=dedent('''
            <white>ZOOM:
            E - zoom in
            T - zoom out
            R - reset
            1 - larger 
                spores
            2 - smaller 
                spores
            ''').strip(),
            position=(-0.75, 0.2),
            scale=0.75,
            color=self.color_manager.get_color('ui', 'text_primary'),
            background=True,
            background_color=self.color_manager.get_color('ui', 'background_transparent'),
        )

        
        self.look_point_text = Text(
            text='LOOK POINT:\nX: 0.000\nZ: 0.000', 
            position=(0.5, 0.45, 0), 
            scale=0.75, 
            color=self.color_manager.get_color('ui', 'text_primary'),
            background=True,
            background_color=self.color_manager.get_color('ui', 'background_transparent'),
            origin=(-0.5, 0.5),
            parent=camera.ui,
            font='VeraMono.ttf',
            font_size=16,
        )


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

        look_point_coords = np.array([x_0, z_0])

        frame_origin_coords = self.scene_setup.frame.origin.real_position
        frame_origin_coords = np.array([frame_origin_coords[0], frame_origin_coords[2]])
        
        current_scale = self.scene_setup.frame.x_axis.scale[0]

        scaled_look_point_coords = (look_point_coords - frame_origin_coords) / current_scale

        x_1 = scaled_look_point_coords[0]
        z_1 = scaled_look_point_coords[1]

        self.look_point_text.text = 'LOOK POINT:\n'
        self.look_point_text.text += f'({x_1:6.3f}, {z_1:6.3f})\n'
        self.look_point_text.text += f'common scale: {current_scale:6.3f}\n'
        self.look_point_text.text += f'spores scale: {self.spores_scale:6.3f}'
        self.look_point_text.background = True

        return x_0, z_0

    def update_transform(self):
        from .link import Link
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

        



